import os
import re
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import RobertaTokenizer


DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")

TOP_K = 5
CHUNK_SIZE = 500
QUERY_SIZE = 64

LANG = os.environ["LANG"]
ROOT_DIR = "../data/cceval_rawdata"
TARGET = f"../data/cceval/{LANG}/line_completion.jsonl"
OUTPUT = f"../data/cceval_custom/cceval-{LANG}-bm25-token.jsonl"

FILE_EXTS = {"python": "py", "java": "java", "typescript": "ts", "csharp": "cs"}

model_name = "microsoft/unixcoder-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)


def main():
    repos = set()
    with open(TARGET) as f:
        for line in f:
            dataset = json.loads(line.strip())
            repos.add(dataset["metadata"]["repository"])

    knowledge_base = []
    error_count = 0
    for repo in tqdm(sorted(repos)):
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
                    "chunk": chunk.split("\n"),
                    "tokens": [t for t in DOCSTRING_REGEX_TOKENIZER.findall(chunk) if t],
                }
                knowledge_base.append(data)

    with open(TARGET) as f, open(OUTPUT, "w") as f_out:
        for line in f:
            task_data = json.loads(line.strip())
            print(task_data["metadata"]["task_id"])

            search_pool_for_repo = []
            for data in knowledge_base:
                if data["repo"] != task_data["metadata"]["repository"] or data["file"] == task_data["metadata"]["file"]:
                    continue
                search_pool_for_repo.append(data)

            prompt_lines = [pl for pl in task_data["prompt"].split("\n") if pl.strip()]
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

            task_data["crossfile_context"]["text"] = crossfile_context
            f_out.write(f"{json.dumps(task_data)}\n")

    print("error_count:", error_count)


if __name__ == "__main__":
    main()
