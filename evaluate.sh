export MODEL="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_NAME=`echo $MODEL | cut -d'/' -f2`

for language in \
    python \
    java \
    csharp; do

    export LANG=$language
    export PROMPT_TYPE=$LANG"-unixcoder-lc2"
    export INPUT="data/cceval_custom/cceval-"$PROMPT_TYPE".jsonl"
    export OUTPUT="output/cceval_"$PROMPT_TYPE"_"$MODEL_NAME".jsonl"

    echo $OUTPUT
    python evaluate.py
    echo 
done

export LANG="python"
for target in \
    line-unixcoder-lc2 \
    api-unixcoder-lc2; do

    export PROMPT_TYPE=$target
    export INPUT="data/repoeval_custom/repoeval-"$PROMPT_TYPE".jsonl"
    export OUTPUT="output/repoeval_"$PROMPT_TYPE"_"$MODEL_NAME".jsonl"
    echo $OUTPUT
    python evaluate.py
    echo 
done
