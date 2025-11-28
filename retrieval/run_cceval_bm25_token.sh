for language in \
    python \
    java \
    csharp \
    typescript; do

    export LANG=$language
    python run_cceval_bm25_token.py

done
