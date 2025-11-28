export CUDA_VISIBLE_DEVICES=0

for target in \
    api_level_completion_1k_context_codegen.test \
    line_level_completion_1k_context_codegen.test; do

    export TARGET=$target
    export LANG="python"
    python run_repoeval_unixcoder_lc2.py

done
