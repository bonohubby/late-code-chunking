export CUDA_VISIBLE_DEVICES=0

for language in \
    python \
    java \
    csharp \
    typescript; do

    export LANG=$language
    python run_cceval_unixcoder_chunk.py

done
