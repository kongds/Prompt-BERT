#!/bin/bash
EXP=$1

function gdrive-get() {
    fileid=$1
    filename=$2
    if [[ "${fileid}" == "" || "${filename}" == "" ]]; then
        echo "gdrive-curl gdrive-url|gdrive-fileid filename"
        return 1
    else
        if [[ ${fileid} = http* ]]; then
            fileid=$(echo ${fileid} | sed "s/http.*drive.google.com.*id=\([^&]*\).*/\1/")
        fi
        echo "Download ${filename} from google drive with id ${fileid}..."
        cookie="/tmp/cookies.txt"
        curl -c ${cookie} -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        confirmid=$(awk '/download/ {print $NF}' ${cookie})
        curl -Lb ${cookie} "https://drive.google.com/uc?export=download&confirm=${confirmid}&id=${fileid}" -o ${filename}
        rm -rf ${cookie}
        return 0
    fi
}

case "$EXP" in
"unsup-bert")
    GDRIVE_CODE=1n9FULUIRBhmhvaSQPaOnsudb_CVZyBli
    ;;
"unsup-roberta")
    GDRIVE_CODE=16qQst04wAr_i59ZL-79CVXoivec4lZOZ
    ;;
"sup-bert")
    GDRIVE_CODE=1TtqYSNeMpzQI59tqu3BNWUbnrkWB4GVm
    ;;
"sup-roberta")
    GDRIVE_CODE=123wpRkpQr3OrlRuM2ZzeId2Mc-uw3ozY
    ;;
*)
esac

if [ ! -d ${EXP} ]; then
    echo "downloading " $EXP
    gdrive-get $GDRIVE_CODE ${EXP}.zip
    unzip ${EXP}.zip
    rm ${EXP}.zip
fi


case "$EXP" in
"unsup-bert")
    CUDA_VISIBLE_DEVICES=0 python evaluation.py \
        --model_name_or_path  $EXP\
        --pooler avg\
        --mode test\
        --mask_embedding_sentence \
        --mask_embedding_sentence_template "*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*"
    ;;
"unsup-roberta")
    CUDA_VISIBLE_DEVICES=0 python evaluation.py \
        --model_name_or_path  $EXP\
        --pooler avg\
        --mode test\
        --mask_embedding_sentence \
        --mask_embedding_sentence_template "*cls*_This_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
    ;;
"sup-bert")
    CUDA_VISIBLE_DEVICES=0 python evaluation.py \
        --model_name_or_path  $EXP\
        --pooler avg\
        --mode test\
        --mask_embedding_sentence \
        --mask_embedding_sentence_org_mlp\
        --mask_embedding_sentence_delta \
        --mask_embedding_sentence_template "*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*"
    ;;
"sup-roberta")
    CUDA_VISIBLE_DEVICES=0 python evaluation.py \
        --model_name_or_path  $EXP\
        --pooler avg\
        --mode test\
        --mask_embedding_sentence \
        --mask_embedding_sentence_use_pooler\
        --mask_embedding_sentence_delta \
        --mask_embedding_sentence_template "*cls*_This_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
    ;;
*)
esac
