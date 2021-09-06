#!/bin/bash
EXP=$1

GPU=0,1,2,3
ES=125 # --eval_steps
BMETRIC=stsb_spearman # --metric_for_best_model
TRAIN_FILE=data/wiki1m_for_simcse.txt
if [[ $EXP == *"eval"* ]]; then
      EXP=`echo $EXP | sed s/eval-//`
      EVAL_ONLY=true
else
      EVAL_ONLY=false
 fi
args=() # flags for training
eargs=() # flags fro evaluation

case "$EXP" in
"roberta-base-embedding-only-remove-baises")
#| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
#| 60.54 | 66.90 | 66.81 | 76.85 | 71.68 |    69.11     |      61.56      | 67.64 |
  EVAL_ONLY=true
  CHECKPOINT=roberta-base
  eargs=(--remove_continue_word\
         --embedding_only)
    ;;
"bert-base-cased-embedding-only-remove-baises")
#| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
#| 57.86 | 68.51 | 66.43 | 75.67 | 68.99 |    64.51     |      60.39      | 66.05 |
  EVAL_ONLY=true
  CHECKPOINT=bert-base-cased
  eargs=(--remove_continue_word\
         --embedding_only)
    ;;
"bert-base-uncased-embedding-only-remove-baises")
#| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
#| 53.09 | 66.48 | 65.09 | 69.80 | 67.85 |    61.60     |      57.80      | 63.10 |
  EVAL_ONLY=true
  CHECKPOINT=bert-base-uncased
  eargs=(--remove_continue_word\
         --embedding_only)
    ;;
"bert-prompt")
#| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
#| 60.96 | 73.83 | 62.18 | 71.54 | 68.68 |    70.60     |      67.16      | 67.85 |
  EVAL_ONLY=true
  CHECKPOINT=bert-base-uncased
  TEMPLATE="*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*"
  eargs=(--mask_embedding_sentence \
         --mask_embedding_sentence_template $TEMPLATE )
    ;;
"bert-optiprompt")
    BC=(python train.py)
    GPU=$2
    BATCH=256
    LR=3e-5
    ES=1000
    EPOCH=5
    TEMPLATE="*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*"
    MODEL=bert-base-uncased
    args=(--mlp_only_train --mask_embedding_sentence\
          --mask_embedding_sentence_template $TEMPLATE\
          --mask_embedding_sentence_autoprompt\
          --mask_embedding_sentence_org_mlp)
    eargs=(--mask_embedding_sentence_autoprompt\
           --mask_embedding_sentence \
           --mask_embedding_sentence_template $TEMPLATE )
    ;;
"unsup-roberta")
    BC=(python train.py)
    GPU=$3
    BATCH=256
    EPOCH=1
    LR=1e-5
    EXP=${EXP}_s$2
    TEMPLATE="*cls*_This_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
    TEMPLATE2="*cls*_The_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
    MODEL=roberta-base
    args=(--mlp_only_train --mask_embedding_sentence\
          --mask_embedding_sentence_template $TEMPLATE\
          --mask_embedding_sentence_different_template $TEMPLATE2\
          --mask_embedding_sentence_delta\
          --seed $2 )
    eargs=(--mask_embedding_sentence \
           --mask_embedding_sentence_template $TEMPLATE )
    ;;
"unsup-bert")
    BC=(python train.py)
    GPU=$3
    BATCH=256
    EPOCH=1
    LR=1e-5
    EXP=${EXP}_s$2
    TEMPLATE="*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*"
    TEMPLATE2="*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*"
    MODEL=bert-base-uncased
    args=(--mlp_only_train --mask_embedding_sentence\
          --mask_embedding_sentence_template $TEMPLATE\
          --mask_embedding_sentence_different_template $TEMPLATE2\
          --mask_embedding_sentence_delta\
          --seed $2 )
    eargs=(--mask_embedding_sentence \
           --mask_embedding_sentence_template $TEMPLATE )
    ;;
"sup-roberta")
    BC=(python -m torch.distributed.launch --nproc_per_node 4 train.py)
    TRAIN_FILE=data/nli_for_simcse.csv
    BATCH=128
    EPOCH=3
    LR=5e-5
    MODEL=roberta-base
    TEMPLATE="*cls*_This_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
    args=(--mask_embedding_sentence\
          --mask_embedding_sentence_template $TEMPLATE\
          --mask_embedding_sentence_delta)
    eargs=(--mask_embedding_sentence_use_pooler\
           --mask_embedding_sentence_delta \
           --mask_embedding_sentence \
           --mask_embedding_sentence_template $TEMPLATE )
    ;;
"sup-bert")
    BC=(python -m torch.distributed.launch --nproc_per_node 4 train.py)
    TRAIN_FILE=data/nli_for_simcse.csv
    BATCH=128
    EPOCH=3
    LR=5e-5
    TEMPLATE="*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*"
    MODEL=bert-base-uncased
    args=(--mask_embedding_sentence\
          --mask_embedding_sentence_template $TEMPLATE\
          --mask_embedding_sentence_delta\
          --mask_embedding_sentence_org_mlp)
    eargs=(--mask_embedding_sentence_org_mlp\
           --mask_embedding_sentence_delta \
           --mask_embedding_sentence \
           --mask_embedding_sentence_template $TEMPLATE )
    ;;
*)
esac

if [ -z "$GPU" ]; then
  GPU=0
fi

if [[ $EVAL_ONLY == false ]]; then
  CHECKPOINT=result/$EXP
  CUDA_VISIBLE_DEVICES=$GPU ${BC[@]}\
              --model_name_or_path $MODEL\
              --train_file $TRAIN_FILE\
              --output_dir $CHECKPOINT\
              --num_train_epochs $EPOCH\
              --per_device_train_batch_size $BATCH \
              --learning_rate $LR \
              --max_seq_length 32\
              --evaluation_strategy steps\
              --metric_for_best_model $BMETRIC\
              --load_best_model_at_end\
              --eval_steps $ES\
              --overwrite_output_dir\
              --temp 0.05\
              --do_train\
              --fp16\
              --preprocessing_num_workers 10\
              ${args[@]}
else
  if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$2
  fi
fi

if [[ $EXP == "sup"* ]]; then
        # rewrite key for supervised model
  python <<EOF
import argparse
import torch
import os
import json
path="$CHECKPOINT"
print("checkpoint -> Huggingface checkpoint for {}".format(path))
state_dict = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=torch.device("cpu"))
new_state_dict = {}
for key, param in state_dict.items():
    if "mlp" in key:
        key = key.replace("mlp", "pooler")
    if "bert." in key:
        key = key.replace("bert.", "")
    if "roberta." in key:
        key = key.replace("roberta.", "")
    new_state_dict[key] = param
torch.save(new_state_dict, os.path.join(path, "pytorch_model.bin"))
config = json.load(open(os.path.join(path, "config.json")))
for i in range(len(config["architectures"])):
    config["architectures"][i] = config["architectures"][i].replace("ForCL", "Model")
json.dump(config, open(os.path.join(path, "config.json"), "w"), indent=2)
EOF
fi

CUDA_VISIBLE_DEVICES=$GPU python evaluation.py \
    --model_name_or_path   $CHECKPOINT \
    --pooler avg\
    --mode test\
    ${eargs[@]}
