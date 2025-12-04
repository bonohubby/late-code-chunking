export MODEL="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_NAME=`echo $MODEL | cut -d'/' -f2`

export LANG="python"
export PROMPT_TYPE=$LANG"-unixcoder-lc2"
export INPUT="data/cceval_custom/cceval-"$PROMPT_TYPE".jsonl"
export OUTPUT="output/cceval_"$PROMPT_TYPE"_"$MODEL_NAME".jsonl"

# export LANG="python"
# export PROMPT_TYPE="line-unixcoder-lc2"
# export INPUT="data/repoeval_custom/repoeval-"$PROMPT_TYPE".jsonl"
# export OUTPUT="output/repoeval_"$PROMPT_TYPE"_"$MODEL_NAME".jsonl"

python generate.py
python evaluate.py
