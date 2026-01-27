# TinyBERT 运行指南 (Learning Version)

这是一个修改后的TinyBERT学习版本，旨在方便调试和查看结果。本项目包含了完整的流程脚本、输入数据（教师/学生模型配置）以及用于存放各阶段结果的文件夹。

## 目录结构说明

以下是`TinyBERT`文件夹中各子文件夹和关键文件的含义：

### 核心代码
- `data_augmentation.py`: **数据增强**脚本。使用GloVe词向量和BERT生成增强数据。
- `pregenerate_training_data.py`: **预生成训练数据**脚本。用于General Distillation阶段的数据准备。
- `general_distill.py`: **通用蒸馏 (General Distillation)** 脚本。让学生模型模仿教师模型的通用语言知识。
- `task_distill.py`: **任务特定蒸馏 (Task-specific Distillation)** 脚本。包含中间层蒸馏和预测层蒸馏。

### 数据与模型文件夹
- **`1_pre_data/`**: 存放预处理和初始数据。
  - `glove.6B.100d.txt`: GloVe 词向量（用于数据增强）。
  - `documents.txt`: 通用语料库（用于通用蒸馏）。
  - `vocab_21128.txt` / `vocab.txt`: 词表文件。
  - `aug.tsv`: 如果运行数据增强，结果会默认输出到这里（或任务文件夹）。
- **`2_pregenerate_data/`**: 存放 `pregenerate_training_data.py` 运行生成的训练数据。
- **`3_general_distill/`**: 存放 `general_distill.py` 训练后的模型（通用蒸馏后的学生模型）。
- **`4_task_distill_mid/`**: 存放 `task_distill.py` 第一阶段（中间层蒸馏）训练后的模型。
- **`5_task_distill_out/`**: 存放 `task_distill.py` 第二阶段（预测层蒸馏）训练后的最终模型。
- **`SST-2/`**: 存放具体任务（这里是SST-2情感分析）的数据集（train, dev, test）。
- **`student_model/`**: 存放初始学生模型的配置 (`config.json`)。
- **`teacher_model/`**: 存放教师模型的配置、词表和权重 (`pytorch_model.bin` 等)。

---

## 运行步骤

所有的运行命令都已经作为注释写在各 Python 脚本的头部。请按照以下顺序执行：
### 0. 下载相关数据
下载 [教师模型](https://huggingface.co/google-bert/bert-base-uncased/tree/main) pytorch_model.bin
大小约440MB，下载到`teacher_model`文件夹下

下载 [GloVe词表](https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/tree/main) glove.6B.100d.txt.gz
大小约134MB，下载并解压到`1_pre_data`文件夹下


### 1. 数据增强 (Data Augmentation)
用于扩充下游任务的数据集。
```bash
python data_augmentation.py --pretrained_bert_model teacher_model --glove_embs 1_pre_data/glove.6B.100d.txt --glue_dir . --task_name SST-2
```

### 2. 预生成训练数据 (Pregenerate Training Data)
将通用语料库转换为用于通用蒸馏的格式。
```bash
python pregenerate_training_data.py --train_corpus 1_pre_data/documents.txt --output_dir 2_pregenerate_data --bert_model teacher_model --do_whole_word_mask
```

### 3. 通用蒸馏 (General Distillation)
使用预生成的通用数据对学生模型进行预训练。
```bash
python general_distill.py --pregenerated_data 2_pregenerate_data --teacher_model teacher_model --student_model student_model --output_dir 3_general_distill --do_lower_case
```

### 4. 任务特定蒸馏 (Task Distillation)

**步骤 4.1: 准备通用蒸馏后的模型**
在运行下一步之前，你需要将上一步输出目录 `3_general_distill` 中的模型文件（例如 `model_epoch_x.bin`）重命名为 `pytorch_model.bin`，以便后续脚本读取。

**步骤 4.2: 中间层蒸馏 (Intermediate Layer Distillation)**
```bash
python task_distill.py --data_dir SST-2 --teacher_model teacher_model --student_model 3_general_distill --task_name sst-2 --output_dir 4_task_distill_mid --aug_train --do_lower_case --pred_distill
```

**步骤 4.3: 预测层蒸馏 (Prediction Layer Distillation)**
使用上一步得到的模型作为输入。
```bash
python task_distill.py --data_dir SST-2 --teacher_model teacher_model --student_model 4_task_distill_mid --task_name sst-2 --output_dir 5_task_distill_out --aug_train --do_lower_case
```
---

**注意**: 运行以上命令前请确保相关依赖已安装，且工作目录位于 `TinyBERT/` 下。

----


========
TinyBERT
========

TinyBERT is 7.5x smaller and 9.4x faster on inference than BERT-base and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages. The overview of TinyBERT learning is illustrated as follows: 
<br />
<br />
<img src="tinybert_overview.png" width="800" height="210"/>
<br />
<br />

For more details about the techniques of TinyBERT, refer to our paper:

[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)


Release Notes
=============
First version: 2019/11/26
Add Chinese General_TinyBERT: 2021.7.27

Installation
============
Run command below to install the environment(**using python3**)
```bash
pip install -r requirements.txt
```

General Distillation
====================
In general distillation, we use the original BERT-base without fine-tuning as the teacher and a large-scale text corpus as the learning data. By performing the Transformer distillation on the text from general domain, we obtain a general TinyBERT which provides a good initialization for the task-specific distillation. 

General distillation has two steps: (1) generate the corpus of json format; (2) run the transformer distillation;

Step 1: use `pregenerate_training_data.py` to produce the corpus of json format  


```
 
# ${BERT_BASE_DIR}$ includes the BERT-base teacher model.
 
python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \ 
                  --bert_model ${BERT_BASE_DIR}$ \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$ 
                             
```

Step 2: use `general_distill.py` to run the general distillation
```
 # ${STUDENT_CONFIG_DIR}$ includes the config file of student_model.
 
python general_distill.py --pregenerated_data ${CORPUS_JSON}$ \ 
                          --teacher_model ${BERT_BASE}$ \
                          --student_model ${STUDENT_CONFIG_DIR}$ \
                          --reduce_memory --do_lower_case \
                          --train_batch_size 256 \
                          --output_dir ${GENERAL_TINYBERT_DIR}$ 
```


We also provide the models of general TinyBERT here and users can skip the general distillation.

=================1st version to reproduce our results in the paper ===========================

[General_TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj) 

[General_TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1wXWR00EHK-Eb7pbyw0VP234i2JTnjJ-x)

=================2nd version (2019/11/18) trained with more (book+wiki) and no `[MASK]` corpus =======

[General_TinyBERT_v2(4layer-312dim)](https://drive.google.com/open?id=1PhI73thKoLU2iliasJmlQXBav3v33-8z)

[General_TinyBERT_v2(6layer-768dim)](https://drive.google.com/open?id=1r2bmEsQe4jUBrzJknnNaBJQDgiRKmQjF)

=================Chinese version trained with WIKI and NEWS corpus =======

[General_TinyBERT_zh(4layer-312dim)](https://huggingface.co/huawei-noah/TinyBERT_4L_zh/tree/main)

[General_TinyBERT_zh(6layer-768dim)](https://huggingface.co/huawei-noah/TinyBERT_6L_zh/tree/main)

Data Augmentation
=================
Data augmentation aims to expand the task-specific training set. Learning more task-related examples, the generalization capabilities of student model can be further improved. We combine a pre-trained language model BERT and GloVe embeddings to do word-level replacement for data augmentation.

Use `data_augmentation.py` to run data augmentation and the augmented dataset `train_aug.tsv` is automatically saved into the corresponding ${GLUE_DIR/TASK_NAME}$
```

python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR}$ \
                            --glove_embs ${GLOVE_EMB}$ \
                            --glue_dir ${GLUE_DIR}$ \  
                            --task_name ${TASK_NAME}$

```
Before running data augmentation of GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory GLUE_DIR. And TASK_NAME can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.

Task-specific Distillation
==========================
In the task-specific distillation, we re-perform the proposed Transformer distillation to further improve TinyBERT by focusing on learning the task-specific knowledge. 

Task-specific distillation includes two steps: (1) intermediate layer distillation; (2) prediction layer distillation.

Step 1: use `task_distill.py` to run the intermediate layer distillation.
```

# ${FT_BERT_BASE_DIR}$ contains the fine-tuned BERT-base model.

python task_distill.py --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${GENERAL_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \ 
                       --output_dir ${TMP_TINYBERT_DIR}$ \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 10 \
                       --aug_train \
                       --do_lower_case  
                         
```


Step 2: use `task_distill.py` to run the prediction layer distillation.
```

python task_distill.py --pred_distill  \
                       --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${TMP_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${TINYBERT_DIR}$ \
                       --aug_train  \  
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --eval_step 100 \
                       --max_seq_length 128 \
                       --train_batch_size 32 
                       
```


We here also provide the distilled TinyBERT(both 4layer-312dim and 6layer-768dim) of all GLUE tasks for evaluation. Every task has its own folder where the corresponding model has been saved.

[TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1_sCARNCgOZZFiWTSgNbE7viW_G5vIXYg) 

[TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1Vf0ZnMhtZFUE0XoD3hTXc6QtHwKr_PwS)


Evaluation
==========================
The `task_distill.py` also provide the evalution by running the following command:

```
${TINYBERT_DIR}$ includes the config file, student model and vocab file.

python task_distill.py --do_eval \
                       --student_model ${TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${OUTPUT_DIR}$ \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128  
                                   
```

To Dos
=========================
* Evaluate TinyBERT on Chinese tasks.
* Tiny*: use NEZHA or ALBERT as the teacher in TinyBERT learning.
* Release better general TinyBERTs.
