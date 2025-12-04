for language in \
    python \
    java \
    csharp; do

    export LANG=$language
    python run_cceval_unixcoder_lc2.py
done
