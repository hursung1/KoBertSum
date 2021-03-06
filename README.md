

# KoBertSum

## 앞서
이 코드는 https://github.com/uoneway/KoBertSum.git 의 fork임을 미리 알립니다.

## Update
[21.07.29.]
- training 수행하는 동안 validation 수행할 수 있도록 코드 수정
  

## 모델 소개

### KoBertSum이란?

KoBERTSUM은 ext 및 abs summarizatoin 분야에서 우수한 성능을 보여주고 있는 [BertSum모델](https://github.com/nlpyang/PreSumm)을 한국어 데이터에 적용할 수 있도록 수정한 한국어 요약 모델입니다.

- Pre-trained BERT로 [KoBERT](https://github.com/SKTBrain/KoBERT)를 이용합니다. 원활한 연결을 위해 [Transformers(](https://github.com/monologg/KoBERT-Transformers)[monologg](https://github.com/monologg/KoBERT-Transformers)[)](https://github.com/monologg/KoBERT-Transformers)를 통해 Huggingface transformers 라이브러리를 사용합니다.

- 이용자가 원하는 데이터도 쉽게 입력 가능합니다.

- `BertSumExt`모델만 지원합니다.



### BertSum이란?

BertSum은 BERT 위에 inter-sentence Transformer 2-layers 를 얹은 구조를 갖습니다. 이를 fine-tuning하여 extract summarization을 수행하는 `BertSumExt`, abstract summarization task를 수행하는 `BertSumAbs` 및 `BertSumExtAbs` 요약모델을 포함하고 있습니다.

- 논문:  [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) (EMNLP 2019 paper)
- 원코드: https://github.com/nlpyang/PreSumm

기 Pre-trained BERT를 summarization task 수행을 위한 embedding layer로 활용하기 위해서는 여러 sentence를 하나의 인풋으로 넣어주고, 각 sentence에 대한 정보를 출력할 수 있도록 입력을 수정해줘야 합니다. 이를 위해

- Input document에서 매 문장의 앞에 [CLS] 토큰을 삽입하고
    ( [CLS] 토큰에 대응하는 BERT 결과값(T[CLS])을 각 문장별 representation으로 간주)

- 매 sentence마다 다른 segment embeddings 토큰을 더해주는 interval segment embeddings을 추가합니다.


## Usage

### Data Preparation
 - `jsonl` 데이터 또는  `.pickle`로 저장한 dataframe 데이터를 바로 처리할 수 있습니다.
 - 파일명에 `train`(필요시 `valid`도), `test` 텍스트가 포함되어 있는지에 따라 데이터 종류를 구분하고 그에 맞게 처리합니다. 따라서 파일명에 해당 키워드가 포함되도록 수정해주세요.
 당연히 `train`(필요시 `valid`도) 데이터는 summary를 포함하고 있어야 합니다.
 - Data 및 폴더 구성은 `datasets/sample`을 참고하세요.

1. 필요 라이브러리 설치하기
    ```
    pip install -r requirements_prepro.txt
    pip install -r requirements.txt
    ```

2. 데이터 추가하기
   - `jsonl` 데이터일 경우, `datasets/` 폴더에 넣어줍니다.
   - `.pickle`로 저장한 dataframe 데이터일 경우, `datasets/df` 폴더에 넣어줍니다.
  
3. 다음 명령어를 실행하여 데이터를 BertSum 모델에 입력 가능한 `.pt` 파일로 변환합니다.   
변환 데이터는 `datasets/bert`에 저장됩니다.
   - 공통사항
     - `dataset`: 데이터셋 이름 (위 2번 단계에서 만들어준 DATASET_NAME과 동일해야 함)
     - `src_name`: 데이터에서 본문에 해당하는 key값(jsonl) 또는 colname(df)
     - `tgt_name`: 데이터에서 요약문에 해당하는 key값(jsonl) 또는 colname(df)
     - `tgt_type`: 데이터 내 요약문이 index list형태로 되어있는지(`idx_list`), string list로 되어 있는지(`str_list`)
     - `train_split_frac`(default=1.0): train 데이터를 train 및 valid 데이터로 분리하고 싶을 때 train 데이터 비율(1이면 분리하지 않음)
     - `n_cpus`(default=2): 연산에 이용할 CPU 수

   -  데이터셋이 `.jsonl`인 경우
    ```
    python make_data.py mode=jsonl_to_bert \
      dataset=DATASET_NAME src_name=SRC_NAME tgt_name=SUMMARY_NAME tgt_type=idx_list  \
      train_split_frac=0.95 \
      n_cpus=3
    ```

   - 데이터셋이 `.pickle`로 저장한 dataframe인 경우
    ```
    python make_data.py mode=df_to_bert \
      dataset=DATASET_NAME src_name=SRC_NAME tgt_name=SUMMARY_NAME tgt_type=idx_list \
      train_split_frac=0.95 \
      n_cpus=3
    ```

  - 한국어 문서 추출요약 AI 경진대회에서 제공된 [Bflysoft-뉴스기사 데이터셋](https://dacon.io/competitions/official/235671/data/)을 사용할 경우
     - 해당 `.jsonl`파일을 다운받아 `datasets/bflysoft_ko` 폴더에 넣어준 후
     - `python make_data.py mode=jsonl_to_bert dataset=bflysoft_ko`를 실행해주면 간편하게 변환할 수 있습니다.
   
  
### Fine-tuning and Inference

1. 필요 라이브러리 설치하기
    ```
    pip install -r requirements.txt
    ```
2. Fine-tuning

    Huggingface를 통해 제공되는 KoBERT 계열 모델을 기반으로 fine-tuning을 진행하고, validation cross-entropy가 더 낮게 나오는 모델의 parameter를 파일로(`.pt`) 저장합니다. validation은 매 500 step마다 수행됩니다.

    - `target_summary_sent`: `abs` 또는 `ext` . 
    - `visible_gpus`: 연산에 이용할 gpu index를 입력. 
      예) (GPU 3개를 이용할 경우): `0,1,2`

    ```
    python main.py -mode train -visible_gpus 0
    ```

    결과는  `ext/models` 폴더 내 finetuning이 실행된 시간을 폴더명으로 가진 폴더에 저장됩니다. 

3. Inference & make submission file

    Validation을 통해 확인한 (cross-entropy 기준) 가장 성능이 우수한 model파일을 불러와 실제로 문서 요약 작업을 수행합니다.

    - `test_from`:  model 파일(`.pt`) 경로
    - `visible_gpus`: 연산에 이용할 gpu index를 입력. 
      예) (GPU 3개를 이용할 경우): `0,1,2`

    ```
    python main.py -mode test -test_from 0909_1800/model_step_25000.pt -visible_gpus 0
    ```

    `ext/data/results/` 폴더에, model의 inference 결과는 `result_0909_1800_step_25000.candidate` 형태로, 정답 파일은 `result_0909_1800_step_25000.gold` 형태로 저장됩니다.
    
    
### Calculate ROUGE Score

1. 한국어 ROUGE package 설치

    [이 링크](https://github.com/hong8e/KoROUGE)의 1번 항목을 따라 KoROUGE package를 설치합니다.

2. ROUGE score 계산하기

    Inference를 통해 생성된 ```.candidate``` 파일과 ```.gold``` 파일이 있는 ```ext/data/results/``` 폴더로 이동하여, ROUGE score를 계산합니다
    ```
    KoROUGE result_0909_1800_step_25000.candidate result_0909_1800_step_25000.gold
    ```
