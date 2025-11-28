# Late Code Chunking (LC-Square) for Repository-Level Code Completion

## Benchmarks for Repository-Level Code Completion
- RepoEval (EMNLP 2023): https://github.com/microsoft/CodeT/tree/main/RepoCoder
- CrossCodeEval (NeurIPS 2023): https://github.com/amazon-science/cceval

## Directory Overview
```
late-code-chunking/
├── evaluate.py
├── generate.py
├── run.sh (run evaluate.py and generate.py)
├── data/ (retrieval results)
│   ├── cceval_custom/
│   └── repoeval_custom/
├── output/ (inference and evaluation results)
│   ├── cceval_python-bm25-lc2_deepseek-coder-1.3b-base.jsonl
│   ├── cceval_python-bm25-lc2_deepseek-coder-1.3b-base_eval.jsonl
│   ├── cceval_python-unixcoder-lc2_deepseek-coder-1.3b-base.jsonl
│   ├── cceval_python-unixcoder-lc2_deepseek-coder-1.3b-base_eval.jsonl
│   ├── ...
└── retrieval/ (retrievals with different chunking strategies)
    ├── run_cceval_unixcoder_chunk.py
    ├── run_cceval_unixcoder_chunk.sh
    ├── run_cceval_unixcoder_function.py
    ├── run_cceval_unixcoder_function.sh
    ├── run_cceval_unixcoder_lc2.py
    ├── run_cceval_unixcoder_lc2.sh
    ├── run_cceval_unixcoder_spilt_aggregate.py
    ├── run_cceval_unixcoder_spilt_aggregate.sh
    ├── run_cceval_unixcoder_window.py
    ├── run_cceval_unixcoder_window.sh
    ├── ...
```

## Experimantal Setup
### Code Retrieval
```bash
# Python 3.12.12
conda install -c pytorch -c nvidia faiss-gpu=1.12.0
pip install tree-sitter tree-sitter-python tree-sitter-java tree-sitter-java tree-sitter-c-sharp tree-sitter-typescript
pip install uv rank_bm25
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Code Completion
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

### Referenced Code
- https://github.com/microsoft/CodeT/tree/main/RepoCoder
- https://github.com/amazon-science/cceval
- https://github.com/microsoft/CodeBERT/tree/master/UniXcoder

## Inference and Evaluation
```bash
./run.sh
```
