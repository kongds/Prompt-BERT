# PromptBERT: Prompt makes BERT Better at Sentence Embeddings

## Overview
We propose PromptBERT, a novel contrastive learning method for learning better sentence representation. We firstly analyze the drawback of current sentence embedding from original BERT and find that it is mainly due to the static token embedding bias and ineffective BERT layers. Then we propose the first prompt-based sentence embeddings method and discuss two prompt representing methods and three prompt searching methods to make BERT achieve better sentence embeddings. Moreover, we propose a novel unsupervised training objective by the technology of template denoising, which substantially shortens the performance gap between the supervised and unsupervised settings. Extensive experiments show the effectiveness of our method. Compared to SimCSE, PromptBert achieves 2.29 and 2.58 points of improvement based on BERT and RoBERTa in the unsupervised setting.
    
## Results on STS Tasks

| Model                                                                                                                    | STS12      | STS13      | STS14      | STS15      | STS16      | STSb       | SICK-R     | Avg.       |
|--------------------------------------------------------------------------------------------------------------------------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| [unsupervised PromptBERT](https://drive.google.com/file/d/1n9FULUIRBhmhvaSQPaOnsudb_CVZyBli/view?usp=sharing)   | 71.56±0.18 | 84.58±0.22 | 76.98±0.26 | 84.47±0.24 | 80.60±0.21 | 81.60±0.22 | 69.87±0.40 | 78.54±0.15 |
| [unsupervised PromptRoBERTa](https://drive.google.com/file/d/16qQst04wAr_i59ZL-79CVXoivec4lZOZ/view?usp=sharing) | 73.94±0.90 | 84.74±0.36 | 77.28±0.41 | 84.99±0.25 | 81.74±0.29 | 81.88±0.37 | 69.50±0.57 | 79.15±0.25 |
| [supervised PromptBERT](https://drive.google.com/file/d/1TtqYSNeMpzQI59tqu3BNWUbnrkWB4GVm/view?usp=sharing)      | 75.48      | 85.59      | 80.57      | 85.99      | 81.08      | 84.56      | 80.52      | 81.97      |
| [supervised PromptRoBERTa](https://drive.google.com/file/d/123wpRkpQr3OrlRuM2ZzeId2Mc-uw3ozY/view?usp=sharing)   | 76.75      | 85.93      | 82.28      | 86.69      | 82.80      | 86.14      | 80.04      | 82.95      |
    
To evaluate the above models, please run the following script, 
```sh
bash eval_only.sh [unsup-bert|unsup-roberta|sup-bert|sup-roberta]
```
## Setup
Install Dependencies

``` sh
pip install -r requirements.txt
```

Download Data

``` sh
cd SentEval/data/downstream/
bash download_dataset.sh
cd -
cd ./data
bash download_wiki.sh
bash download_nli.sh
cd -
```

## Prompt with unfine-tuned BERT

bert-base-uncased with prompt

``` sh
./run.sh bert-prompt
```

bert-base-uncased with optiprompt

``` sh
./run.sh bert-optiprompt
```

## Train PromptBERT and PromptRoBERTa
unsupervised

``` sh
SEED=0
./run.sh unsup-roberta $SEED
```

``` sh
SEED=0
./run.sh unsup-bert $SEED
```
supervised

``` sh
./run.sh sup-roberta 
```

``` sh
./run.sh sup-bert
```

 
## Static token embedding with removing embedding biases
robert-base, bert-base-cased and robert-base-uncased
```sh
./run.sh roberta-base-static-embedding-remove-baises
./run.sh bert-base-cased-static-embedding-remove-baises
./run.sh bert-base-uncased-static-embedding-remove-baises
```
## Calculation of anisotropy
To calculate the sentence anisotropy

```sh
EXP=bert-base-uncased | bert-base-uncased-static-embedding | bert-base-uncased-static-embedding-remove-baises 
./run.sh calc-anisotropy-${EXP}
```

## Acknowledgement
Our Code is based on SimCSE
