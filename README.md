# Late Code Chunking (LC-Square) for Repository-Level Code Completion

## Benchmarks for Repository-Level Code Completion
- RepoEval (EMNLP 2023): https://github.com/microsoft/CodeT/tree/main/RepoCoder
- CrossCodeEval (NeurIPS 2023): https://github.com/amazon-science/cceval

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
