# Late Code Chunking (LC-Square) for Repository-Level Code Completion

## Benchmarks for Repository-Level Code Completion
- RepoEval (EMNLP 2023): https://github.com/microsoft/CodeT/tree/main/RepoCoder
- CrossCodeEval (NeurIPS 2023): https://github.com/amazon-science/cceval

## Directory Overview
```bash
late-code-chunking/
├── evaluate.py
├── evaluate.sh
├── generate.py
├── generate.sh
├── data/
│   ├── cceval/           # CrossCodeEval benchmark
│   │   ├── python/
│   │   ├── ...
│   ├── cceval_custom/    # Retrieval results
│   │   ├── cceval-python-unixcoder-lc2.jsonl
│   │   ├── ...
│   ├── cceval_rawdata/   # CrossCodeEval dataset
│   │   ├── 0x80-isolate-package-4fe8eaf/
│   │   ├── ...
│   ├── repoeval/         # RepoEval benchmark
│   │   ├── api_level_completion_1k_context_codegen.test.jsonl
│   │   ├── ...
│   ├── repoeval_custom/  # Retrieval results
│   │   ├── repoeval-line-unixcoder-lc2.jsonl
│   │   ├── ...
│   ├── repoeval_rawdata/ # RepoEval dataset
│   │   ├── alibaba_FederatedScope/
│   │   ├── ...
├── output/               # Inference and evaluation results
│   ├── cceval_python-unixcoder-lc2_deepseek-coder-1.3b-base.jsonl
│   ├── cceval_python-unixcoder-lc2_deepseek-coder-1.3b-base_eval.jsonl
│   ├── ...
└── retriever/            # Retrievers with different chunking strategies
    ├── run_cceval_unixcoder_lc2.py
    ├── run_cceval_unixcoder_lc2.sh
    ├── ...
```

## Experimantal Setup
### Retrieval
```bash
# Python 3.12.12
conda install -c pytorch -c nvidia faiss-gpu=1.12.0
pip install tree-sitter tree-sitter-python tree-sitter-java tree-sitter-java tree-sitter-c-sharp
pip install uv rank_bm25
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Code Completion and Evaluation
```bash
# Python 3.12.12
pip install thefuzz
pip install uv
uv pip install vllm --torch-backend=cu128
```

### Evaluated Models
- deepseek-coder-1.3b-base
- deepseek-coder-6.7b-base
- starcoder2-3b
- starcoder2-7b
- codegemma-7b
- Codestral-22B-v0.1

### Evaluation
```bash
./evaluate.sh

output/cceval_python-unixcoder-lc2_deepseek-coder-1.3b-base.jsonl
Total:2664, EM Count:702
EM: 26.35
ES: 71.92

output/cceval_java-unixcoder-lc2_deepseek-coder-1.3b-base.jsonl
Total:2139, EM Count:518
EM: 24.22
ES: 66.79

output/cceval_csharp-unixcoder-lc2_deepseek-coder-1.3b-base.jsonl
Total:1768, EM Count:384
EM: 21.72
ES: 68.19

output/repoeval_line-unixcoder-lc2_deepseek-coder-1.3b-base.jsonl
Total:1600, EM Count:722
EM: 45.12
ES: 72.87

output/repoeval_api-unixcoder-lc2_deepseek-coder-1.3b-base.jsonl
Total:1600, EM Count:720
EM: 45.0
ES: 74.48
```

## Referenced Code
- https://github.com/microsoft/CodeT/tree/main/RepoCoder
- https://github.com/amazon-science/cceval
- https://github.com/microsoft/CodeBERT/tree/master/UniXcoder
