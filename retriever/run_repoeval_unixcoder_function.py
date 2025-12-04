import os
import json
import faiss
import torch
from glob import glob
from tqdm import tqdm
from model import Model
from transformers import RobertaModel, RobertaTokenizer
from treesitter import get_functions


TOP_K = 5
CHUNK_SIZE = 500
QUERY_SIZE = 64

LANG = os.environ["LANG"]
ROOT_DIR = "../data/repoeval_rawdata"
TARGET = f"../data/repoeval/{os.environ['TARGET']}.jsonl"
OUTPUT = f"../data/repoeval_custom/repoeval-{os.environ['TARGET'].split('_')[0]}-unixcoder-function.jsonl"

FILE_EXTS = {"python": "py", "java": "java", "typescript": "ts", "csharp": "cs"}

model_name = "microsoft/unixcoder-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
model = Model(model)
device = torch.device("cuda")
model.to(device)
code_length = 512


def embed(model, tokenizer, device, code_length, code):
    code_tokens = tokenizer.tokenize(code)[: code_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length
    code_inputs = torch.tensor([code_ids]).to(device)
    code_vec = model(code_inputs)
    return code_vec.cpu().numpy()


def main():
    repos = set()
    with open(TARGET) as f:
        for line in f:
            data = json.loads(line.strip())
            repos.add(data["metadata"]["fpath_tuple"][0])

    indices = {}
    gpu_res = faiss.StandardGpuResources()
    model.eval()
    error_count = 0
    with torch.no_grad():
        for repo in tqdm(sorted(repos)):
            index = faiss.IndexFlatIP(768)
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            indices[repo] = {
                "index": gpu_index,
                "data": [],
            }
            src_paths = glob(os.path.join(ROOT_DIR, repo, f"**/*.{FILE_EXTS[LANG]}"), recursive=True)
            for src_path in src_paths:
                if not os.path.isfile(src_path):
                    continue

                with open(src_path) as f:
                    try:
                        doc = f.read()
                    except UnicodeDecodeError:
                        error_count += 1
                        continue

                lines = [line for line in doc.split("\n") if line.strip()]
                if not lines:
                    continue

                definitions = get_functions(src_path)
                for def_name, def_signature, def_body in definitions:
                    chunk = def_body
                    embedding = embed(model, tokenizer, device, code_length, chunk)
                    indices[repo]["index"].add(embedding)
                    indices[repo]["data"].append(
                        {
                            "file": src_path[len(ROOT_DIR) + len(repo) + 2 :],
                            "cc": chunk.split("\n"),
                        }
                    )

        with open(TARGET) as f, open(OUTPUT, "w") as f_out:
            for line in f:
                task_data = json.loads(line.strip())
                print(task_data["metadata"]["task_id"])

                task_data["groundtruth"] = task_data["metadata"]["ground_truth"]
                task_data["metadata"]["repository"] = task_data["metadata"]["fpath_tuple"][0]
                task_data["metadata"]["file"] = "/".join(task_data["metadata"]["fpath_tuple"][1:])

                repo = task_data["metadata"]["repository"]
                file = task_data["metadata"]["file"]

                with open(os.path.join(ROOT_DIR, repo, file)) as f_p:
                    fpath = f_p.read()
                fpath = fpath.split("\n")[task_data["metadata"]["context_start_lineno"] : task_data["metadata"]["line_no"]]

                prompt_lines = [pl for pl in fpath if pl.strip()]
                query = "\n".join(prompt_lines)
                ids = tokenizer.encode(query, add_special_tokens=False)
                query = tokenizer.decode(ids[-QUERY_SIZE:])
                embedding = embed(model, tokenizer, device, code_length, query)

                task_data["crossfile_context"] = {}
                task_data["crossfile_context"]["list"] = []
                crossfile_context = "# Here are some relevant code fragments from other files of the repo:\n\n"

                retrieved = 0
                D, I = indices[repo]["index"].search(embedding, 2048)
                for k, i in enumerate(I[0]):
                    if indices[repo]["data"][i]["file"] == file:
                        continue

                    retrieved += 1
                    task_data["crossfile_context"]["list"].append(
                        {
                            "retrieved_chunk": "\n".join(indices[repo]["data"][i]["cc"]),
                            "filename": indices[repo]["data"][i]["file"],
                            "score": D[0][k].astype(float),
                        }
                    )
                    crossfile_context += "# the below code fragment can be found in:\n"
                    crossfile_context += f"# {indices[repo]['data'][i]['file']}\n"
                    crossfile_context += ("\n".join([f"# {c}" for c in indices[repo]["data"][i]["cc"]]) + "\n\n")
                    if TOP_K == retrieved:
                        task_data["crossfile_context"]["text"] = crossfile_context
                        f_out.write(f"{json.dumps(task_data)}\n")
                        break

        print("error_count:", error_count)


if __name__ == "__main__":
    main()
