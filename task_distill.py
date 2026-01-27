# coding=utf-8
# 2019.12.2-Changed for TinyBERT task-specific distillation
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
先将 3_general_distill 目录下某个 .bin 模型文件改名为 pytorch_model.bin

终端运行命令，蒸馏中间层
--data_dir SST-2 --teacher_model teacher_model --student_model 3_general_distill --task_name sst-2 --output_dir 4_task_distill_mid --aug_train --do_lower_case --pred_distill
在运行命令，蒸馏输出
--data_dir SST-2 --teacher_model teacher_model --student_model 4_task_distill_mid --task_name sst-2 --output_dir 5_task_distill_out --aug_train --do_lower_case
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from numpy.compat import unicode
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

csv.field_size_limit(100*1024*1024)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False


class InputExample(object):
    """单条样本的原始文本表示."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """构建一条样本的原始输入.

        Args:
            guid: 样本唯一ID.
            text_a: 句子A原始文本.
            text_b: 句子B原始文本(可选).
            label: 监督标签(训练/验证时使用).
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """单条样本的数值特征(已tokenize+padding)."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        # input_ids/input_mask/segment_ids: 长度为max_seq_length的列表
        # label_id: 分类为int，回归为float
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


class DataProcessor(object):
    """任务数据处理基类: 读取TSV并转换为InputExample."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # 读取TSV文件, 返回行列表(每行是字段list)
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """MRPC数据集处理: 句子对二分类."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # MRPC列: [label, id1, id2, s1, s2]
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """MNLI数据集处理: 蕴含三分类."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # MNLI列: 取第8/9列为文本, 最后一列为标签
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """MNLI-mismatched验证集处理."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """CoLA数据集处理: 句子可接受性二分类."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # CoLA列: [id, label, _, sentence]
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """SST-2情感分类处理."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # SST-2列: [sentence, label]
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """STS-B语义相似度回归处理."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # STS-B列: 取第7/8列为文本, 最后一列为浮点分数
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """QQP问题对二分类处理."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # QQP列: [id, qid1, qid2, q1, q2, label]
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """QNLI句子对蕴含二分类处理."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # QNLI列: [id, question, sentence, label]
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """RTE文本蕴含二分类处理."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # RTE列: [id, premise, hypothesis, label]
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """WNLI代词消歧二分类处理."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # WNLI列: [id, sentence1, sentence2, label]
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    # 将InputExample转换为模型输入特征
    # 输入: examples(文本), label_list, max_seq_length, tokenizer
    # 输出: InputFeatures列表, 每条包含定长input_ids/input_mask/segment_ids/label_id

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 对文本进行WordPiece分词
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # 句子对任务需要为[CLS]/[SEP]留位置
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # 组装BERT输入格式: [CLS] A [SEP] (B [SEP])
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        # 转为id，并构造mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        # padding到固定长度
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # 标签数值化: 分类为int，回归为float
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # 逐步截断较长序列，直到总长度<=max_length
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_tensor_data(output_mode, features):
    # 将特征列表转成TensorDataset
    # 输出tensor形状:
    # all_input_ids/all_input_mask/all_segment_ids: [num_examples, max_seq_length]
    # all_label_ids: [num_examples]
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def result_to_file(result, file_name):
    # 将评估结果写入文件，便于训练过程追踪
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    # 在验证集上评估模型
    # 输入: model, eval_dataloader(包含input_ids等), output_mode, eval_labels
    # 输出: result字典(含eval_loss和各任务指标)
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            # logits: 分类任务[batch_size, num_labels], 回归任务[batch_size, 1]
            logits, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # added arguments
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)
    parser.add_argument('--pred_distill',
                        action='store_true')
    parser.add_argument('--data_url',
                        type=str,
                        default="")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # 任务名 -> Processor映射
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor
    }

    # 任务名 -> 输出类型(分类/回归)
    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification"
    }

    # intermediate distillation default parameters
    # 任务蒸馏常用默认参数
    default_params = {
        "cola": {"num_train_epochs": 50, "max_seq_length": 64},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
        "sst-2": {"num_train_epochs": 10, "max_seq_length": 64},
        "sts-b": {"num_train_epochs": 20, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 20, "max_seq_length": 128}
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    # 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 任务设置与输出目录
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name in default_params:
        args.max_seq_len = default_params[task_name]["max_seq_length"]

    if not args.pred_distill and not args.do_eval:
        if task_name in default_params:
            args.num_train_epoch = default_params[task_name]["num_train_epochs"]

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    # 初始化Processor与标签空间
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # 使用学生模型的词表(与学生模型配置一致)
    """
    通用蒸馏：学生还没入校，，一切按教师的标准来，词表直接用教师的。
    任务蒸馏：学生已经成型，针对这个学生进行任务优化，用它自带的工具。
    根据特定任务可以进行词表修剪，删去不用的词
    """
    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if not args.do_eval:
        # 训练集不增强
        if not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir)
        else:
            train_examples = processor.get_aug_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        # 梯度累积折算batch
        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        # 训练总步数
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        # 构建训练特征与DataLoader
        """重点，加载训练集，文本转token"""
        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)
        train_data, _ = get_tensor_data(output_mode, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # 验证集特征与DataLoader
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    """重点，加载教师、学生模型"""
    if not args.do_eval:
        # 教师模型用于蒸馏
        teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
        teacher_model.to(device)

    # 学生模型用于训练/评估
    student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model, num_labels=num_labels)
    student_model.to(device)
    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(student_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)
        # 准备优化器与参数分组
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # pred_distill仅蒸馏logits时不需要warmup
        schedule = 'warmup_linear'
        if not args.pred_distill:
            schedule = 'none'
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        # 准备损失函数
        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            # 用教师soft label做交叉熵(蒸馏logits)
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)

                # 输入张量:
                # input_ids/input_mask/segment_ids: [batch_size, seq_len]
                # label_ids: [batch_size]
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.

                # 学生模型输出:
                # student_logits: [batch_size, num_labels]或[batch_size, 1]
                # student_atts: 每层注意力 [batch_size, num_heads, seq_len, seq_len]
                # student_reps: embedding+每层输出 [batch_size, seq_len, hidden]
                student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask,
                                                                           is_student=True)

                with torch.no_grad():
                    # 教师模型输出用于蒸馏(不反传)
                    teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

                if not args.pred_distill:
                    # 中间层蒸馏: 对齐注意力与隐藏状态
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]
                    # 对齐注意力层
                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                  teacher_att)

                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss
                    # 对齐表征层
                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps
                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss

                    # 总损失=注意力MSE+表示MSE
                    """重点，第一次任务蒸馏 计算中间层损失函数"""
                    loss = rep_loss + att_loss
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()
                else:
                    # 预测蒸馏: 仅对齐logits
                    if output_mode == "classification":
                        cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                      teacher_logits / args.temperature)
                    elif output_mode == "regression":
                        loss_mse = MSELoss()
                        cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                    """重点，第二次任务蒸馏 计算输出损失函数"""
                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # 反向传播
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # 参数更新
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % args.eval_step == 0:
                    # 周期性评估与保存
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    loss = tr_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    result = {}
                    if args.pred_distill:
                        result = do_eval(student_model, task_name, eval_dataloader,
                                         device, output_mode, eval_labels, num_labels)
                    result['global_step'] = global_step
                    result['cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['loss'] = loss

                    result_to_file(result, output_eval_file)

                    if not args.pred_distill:
                        save_model = True
                    else:
                        save_model = False

                        if task_name in acc_tasks and result['acc'] > best_dev_acc:
                            best_dev_acc = result['acc']
                            save_model = True

                        if task_name in corr_tasks and result['corr'] > best_dev_acc:
                            best_dev_acc = result['corr']
                            save_model = True

                        if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                            best_dev_acc = result['mcc']
                            save_model = True

                    if save_model:
                        logger.info("***** Save model *****")

                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                        model_name = WEIGHTS_NAME
                        # if not args.pred_distill:
                        #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                        # Test mnli-mm
                        if args.pred_distill and task_name == "mnli":
                            # 额外评估mnli-mm并输出到单独目录
                            task_name = "mnli-mm"
                            processor = processors[task_name]()
                            if not os.path.exists(args.output_dir + '-MM'):
                                os.makedirs(args.output_dir + '-MM')

                            eval_examples = processor.get_dev_examples(args.data_dir)

                            eval_features = convert_examples_to_features(
                                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
                            eval_data, eval_labels = get_tensor_data(output_mode, eval_features)

                            logger.info("***** Running mm evaluation *****")
                            logger.info("  Num examples = %d", len(eval_examples))
                            logger.info("  Batch size = %d", args.eval_batch_size)

                            eval_sampler = SequentialSampler(eval_data)
                            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                                         batch_size=args.eval_batch_size)

                            result = do_eval(student_model, task_name, eval_dataloader,
                                             device, output_mode, eval_labels, num_labels)

                            result['global_step'] = global_step

                            tmp_output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
                            result_to_file(result, tmp_output_eval_file)

                            task_name = 'mnli'

                        if oncloud:
                            logging.info(mox.file.list_directory(args.output_dir, recursive=True))
                            logging.info(mox.file.list_directory('.', recursive=True))
                            mox.file.copy_parallel(args.output_dir, args.data_url)
                            mox.file.copy_parallel('.', args.data_url)

                    student_model.train()


if __name__ == "__main__":
    main()
