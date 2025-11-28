export MODEL="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_NAME=`echo $MODEL | cut -d'/' -f2`

export LANG="python"
export PROMPT_TYPE=$LANG"-bm25-lc2"

export OUTPUT="output/cceval_"$PROMPT_TYPE"_"$MODEL_NAME".jsonl"

python evaluate.py
