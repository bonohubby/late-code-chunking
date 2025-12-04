import os
import re
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import RobertaTokenizer
from treesitter import get_functions, get_function_calls


DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")

TOP_K = 5
CHUNK_SIZE = 500
QUERY_SIZE = 64

LANG = os.environ["LANG"]
ROOT_DIR = "../data/repoeval_rawdata"
TARGET = f"../data/repoeval/{os.environ['TARGET']}.jsonl"
OUTPUT = f"../data/repoeval_custom/repoeval-{os.environ['TARGET'].split('_')[0]}-bm25-lc2.jsonl"

FILE_EXTS = {"python": "py", "java": "java", "typescript": "ts", "csharp": "cs"}

model_name = "microsoft/unixcoder-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)


def main():
    repos = set()
    with open(TARGET) as f:
        for line in f:
            data = json.loads(line.strip())
            repos.add(data["metadata"]["fpath_tuple"][0])

    knowledge_base = []
    def_kb = {}
    error_count = 0
    for repo in tqdm(sorted(repos)):
        def_kb[repo] = {}
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

            doc = "\n".join(lines)
            ids = tokenizer.encode(doc, add_special_tokens=False)
            for idx in range(0, len(ids), CHUNK_SIZE):
                chunk = tokenizer.decode(ids[idx : idx + CHUNK_SIZE])
                data = {
                    "repo": repo,
                    "file": src_path[len(ROOT_DIR) + len(repo) + 2 :],
                    "chunk": tokenizer.decode(ids[idx : idx + CHUNK_SIZE * 2]).split("\n"),
                    "chunk_for_ast": chunk.split("\n"),
                    "tokens": [t for t in DOCSTRING_REGEX_TOKENIZER.findall(chunk) if t],
                }
                knowledge_base.append(data)

            definitions = get_functions(src_path)
            for def_name, def_body in definitions:
                def_kb[repo][def_name] = {
                "file": src_path[len(ROOT_DIR) + len(repo) + 2:],
                "function": def_body.split("\n"),
            }

    with open(TARGET) as f, open(OUTPUT, "w") as f_out:
        for line in f:
            task_data = json.loads(line.strip())

            task_data["groundtruth"] = task_data["metadata"]["ground_truth"]
            task_data["metadata"]["repository"] = task_data["metadata"]["fpath_tuple"][0]
            task_data["metadata"]["file"] = "/".join(task_data["metadata"]["fpath_tuple"][1:])
            print(task_data["metadata"]["task_id"])

            repo = task_data["metadata"]["repository"]
            file = task_data["metadata"]["file"]

            search_pool_for_repo = []
            for data in knowledge_base:
                if data["repo"] != task_data["metadata"]["repository"] or data["file"] == task_data["metadata"]["file"]:
                    continue
                search_pool_for_repo.append(data)

            with open(os.path.join(ROOT_DIR, task_data["metadata"]["repository"], task_data["metadata"]["file"])) as f_p:
                fpath = f_p.read()
            fpath = fpath.split("\n")[task_data["metadata"]["context_start_lineno"] : task_data["metadata"]["line_no"]]

            prompt_lines = [pl for pl in fpath if pl.strip()]
            query = "\n".join(prompt_lines)
            ids = tokenizer.encode(query, add_special_tokens=False)
            query = tokenizer.decode(ids[-QUERY_SIZE:])
            query = [t for t in DOCSTRING_REGEX_TOKENIZER.findall(query) if t]

            bm25 = BM25Okapi([data["tokens"] for data in search_pool_for_repo])
            scores = bm25.get_scores(query)
            top_n = np.argsort(scores)[::-1][:TOP_K]

            task_data["crossfile_context"] = {}
            task_data["crossfile_context"]["list"] = []
            crossfile_context = "# Here are some relevant code fragments from other files of the repo:\n\n"
            called = []
            for idx in top_n:
                task_data["crossfile_context"]["list"].append(
                    {
                        "retrieved_chunk": "\n".join(search_pool_for_repo[idx]["chunk"]),
                        "filename": search_pool_for_repo[idx]["file"],
                        "score": scores[idx],
                    }
                )
                crossfile_context += "# the below code fragment can be found in:\n"
                crossfile_context += f"# {search_pool_for_repo[idx]['file']}\n"
                crossfile_context += "\n".join([f"# {c}" for c in search_pool_for_repo[idx]["chunk"]]) + "\n\n"

                for call in get_function_calls(search_pool_for_repo[idx]["chunk_for_ast"]):
                    if call not in def_kb[repo]:
                        continue

                    if call in called:
                        continue
                    
                    if def_kb[repo][call]["file"] == file:
                        continue

                    crossfile_context += "\n".join([f"# {c}" for c in def_kb[repo][call]["function"]]) + "\n\n"

            task_data["crossfile_context"]["text"] = crossfile_context
            f_out.write(f"{json.dumps(task_data)}\n")

    print("error_count:", error_count)


if __name__ == "__main__":
    main()
