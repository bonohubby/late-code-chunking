# Late Code Chunking: A Code Chunking Strategy for Repository-Level Code Completion

## Paper

- ACL 2026 Main: [Late Code Chunking: A Code Chunking Strategy for Repository-Level Code Completion](https://aclanthology.org/2026.acl-short.64/)

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

## Citation

```bibtex
@inproceedings{oh-lee-2026-late,
    title = "Late Code Chunking: A Code Chunking Strategy for Repository-Level Code Completion",
    author = "Oh, Seungmin  and
      Lee, Eunseok",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-short.64/",
    pages = "780--786",
    ISBN = "979-8-89176-391-3",
    abstract = "This paper introduces Late Code Chunking (LC$^2$), a chunking strategy designed to improve the semantic understanding of code segments for Large Language Models (LLMs). Repository-level code completion requires predicting the completion of unfinished code by leveraging cross-file context spread across a repository. However, when retrieved fragments have missing semantics{---}the loss of structural or behavioral information during chunking{---}LLMs struggle to interpret the target code. To address this, LC$^2$ refines retrieved chunks by constructing a dual context: a ``Code Retrieval Context'' optimized for similarity-based search, and a ``Code Comprehension Context'' that serves as a late enrichment step through context expansion and augmentation. This dual-context design reduces information loss due to chunking and enhances the ability of LLMs to utilize retrieved code. Additionally, we introduce an Asymmetric Query-Chunk Sizing strategy to further optimize retrieval quality by minimizing query noise. Our experiments demonstrate that LC$^2$ provides robust performance gains, achieving a statistically significant 19.7{\%} improvement in Exact Match accuracy on the CrossCodeEval benchmark compared to the best existing chunking method."
}
```
