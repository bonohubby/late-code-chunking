#!/bin/bash  

export CUDA_VISIBLE_DEVICES="0"

for language in \
    python; do

    export LANG=$language

    for prompt in \
        nomic-long; do

        export PROMPT_TYPE=$LANG"-"$prompt

        for model in \
            deepseek-ai/deepseek-coder-1.3b-base; do

            export MODEL=$model
            MODEL_NAME=`echo $MODEL | cut -d'/' -f2`

            export INPUT="data/cceval_custom/cceval-"$PROMPT_TYPE".jsonl"

            MODEL_NAME=`echo $MODEL | cut -d'/' -f2`
            export OUTPUT="output/cceval_"$PROMPT_TYPE"_"$MODEL_NAME".jsonl"

            python generate.py

            RESULT_NAME="result_0.txt"
            echo $MODEL $PROMPT_TYPE $INPUT >> $RESULT_NAME
            python evaluate.py >> $RESULT_NAME
            echo '' >> $RESULT_NAME
        done
    done
done
