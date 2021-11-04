# Prompt-BERT: Prompt makes BERT Better at Sentence Embeddings

    
## Results on STS Tasks

| Model                                                                                                                    | STS12 | STS13 | STS14 | STS15 | STS16 | STSb  | SICK-R | Avg.  |
|--------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|
| unsup-prompt-bert-base  [Download](https://drive.google.com/file/d/1n9FULUIRBhmhvaSQPaOnsudb_CVZyBli/view?usp=sharing)   | 71.98 | 84.66 | 77.13 | 84.52 | 81.10 | 82.03 | 70.64  | 78.87 |
| unsup-prompt-roberta-base [Download](https://drive.google.com/file/d/16qQst04wAr_i59ZL-79CVXoivec4lZOZ/view?usp=sharing) | 73.98 | 84.73 | 77.88 | 84.93 | 81.89 | 82.74 | 69.21  | 79.34 |
| sup-prompt-bert-base [Download](https://drive.google.com/file/d/1TtqYSNeMpzQI59tqu3BNWUbnrkWB4GVm/view?usp=sharing)      | 75.48 | 85.59 | 80.57 | 85.99 | 81.08 | 84.56 | 80.52  | 81.97 |
| sup-prompt-roberta-base [Download](https://drive.google.com/file/d/123wpRkpQr3OrlRuM2ZzeId2Mc-uw3ozY/view?usp=sharing)   | 76.75 | 85.93 | 82.28 | 86.69 | 82.80 | 86.14 | 80.04  | 82.95 |
    
## Download Data

``` sh
cd SentEval/data/downstream/
bash download_dataset.sh
cd -
cd ./data
bash download_wiki.sh
bash download_nli.sh
cd -
```

## Static token embedding with removing embedding biases
robert-base, bert-base-cased and robert-base-uncased
```sh
./run.sh roberta-base-embedding-only-remove-baises
./run.sh bert-base-cased-embedding-only-remove-baises
./run.sh bert-base-uncased-embedding-only-remove-baises
```

## Non fine-tuned BERT with Prompt

bert-base-uncased with prompt
``` sh
./run.sh bert-prompt
```

bert-base-uncased with optiprompt
``` sh
./run.sh bert-optiprompt
```

## fine-tuned BERT with Prompt
### unsupervised

``` sh
SEED=0
./run.sh unsup-roberta $SEED
```

``` sh
SEED=0
./run.sh unsup-bert $SEED
```
### supervised

``` sh
./run.sh sup-roberta 
```

``` sh
./run.sh sup-bert
```

Our Code is based on SimCSE
