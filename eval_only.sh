#!/bin/bash
EXP=$1

case "$EXP" in
"unsup-bert")
    MODEL=royokong/unsup-PromptBERT
    ;;
"unsup-roberta")
    MODEL=royokong/unsup-PromptRoBERTa
    ;;
"sup-bert")
    MODEL=royokong/sup-PromptBERT
    ;;
"sup-roberta")
    MODEL=royokong/sup-PromptRoBERTa
    ;;
*)
esac

case "$EXP" in
"unsup-bert")
    CUDA_VISIBLE_DEVICES=0 python evaluation.py \
        --model_name_or_path  $MODEL\
        --pooler avg\
        --mode test\
        --mask_embedding_sentence \
        --mask_embedding_sentence_template "*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*"
    ;;
"unsup-roberta")
    CUDA_VISIBLE_DEVICES=0 python evaluation.py \
        --model_name_or_path  $MODEL\
        --pooler avg\
        --mode test\
        --mask_embedding_sentence \
        --mask_embedding_sentence_template "*cls*_This_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
    ;;
"sup-bert")
    CUDA_VISIBLE_DEVICES=0 python evaluation.py \
        --model_name_or_path  $MODEL\
        --pooler avg\
        --mode test\
        --mask_embedding_sentence \
        --mask_embedding_sentence_org_mlp\
        --mask_embedding_sentence_delta \
        --mask_embedding_sentence_template "*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*"
    ;;
"sup-roberta")
    CUDA_VISIBLE_DEVICES=0 python evaluation.py \
        --model_name_or_path  $MODEL\
        --pooler avg\
        --mode test\
        --mask_embedding_sentence \
        --mask_embedding_sentence_use_pooler\
        --mask_embedding_sentence_delta \
        --mask_embedding_sentence_template "*cls*_This_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
    ;;
*)
esac
