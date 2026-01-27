# coding=utf-8
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
终端运行命令
--train_corpus 1_pre_data/documents.txt --output_dir 2_pregenerate_data --bert_model teacher_model --do_whole_word_mask
"""

import json
import collections
import logging
import os
import shelve
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
from multiprocessing import Pool

import numpy as np
from random import random, randrange, randint, shuffle, choice

from transformer.tokenization import BertTokenizer


# This is used for running on Huawei Cloud.
oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        # 文档缓存: 既可在内存保存，也可用磁盘shelve降低内存占用
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open('/cache/shelf.db',
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        """document是句子list，每个句子已tokenize为token列表"""
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        """计算总文档长度的累积和，用于按句子数加权采样"""
        self.doc_cumsum = np.cumsum(self.doc_lengths)   # 一维数组累加，cumsum[1,2,3,4] = [1,3,6,10]
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        """使用当前迭代计数器确保不会重复采样同一个文档"""
        if sentence_weighted:
            # 通过句子加权，我们按句子长度比例抽样文档
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()        # doc_cumsum=[1,3,6,10]
            rand_start = self.doc_cumsum[current_idx]   # doc_cumsum[2]=3,取长度为3的句子,rand_start=6
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx] # rand_end=6+10-3=13
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max  # 只取到[1,2,3,4]中1，2，4长度的句子，不会抽到重复的current_idx
            # np.searchsorted假设插入一个数，获取其会插入的位置id。
            # 因为是带句子长度加权的，所以sentence_index可能不是句子开头，而是句中id，借助np.searchsorted获取该id在原文档的idx
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # 如果不使用句子权重，那么每个文档都有同等的被选中机会
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """
    将一对序列截断为最大序列长度。摘自谷歌的BERT仓库
    非常普通的方法
    """
    # 逐步截断较长序列，直到总长度<=max_num_tokens
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # 有时从前端截断，有时从后是为了增加更多随机性，避免偏见。
        if random() < 0.5:
            del trunc_tokens[0]     # 删前, O(n)
        else:
            trunc_tokens.pop()      # 删后, O(1)


# 掩码预测任务，封装方法MaskedLmInstance，用于掩盖词并记录真是标签。如 masked_instance = MaskedLmInstance(index=2, label="brown")
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """
    为掩码语言模型目标创建预测。这大多是从 Google BERT 仓库复制的，但经过多次重构来清理并剔除许多不必要的变量。
    输入: tokens(含[CLS]/[SEP]), masked_lm_prob(掩码概率), max_predictions_per_seq(单序列最大预测掩码数), whole_word_mask(整词掩码开关), vocab_list(词表列表)
    输出: 被mask后的tokens, mask_indices(被mask位置索引), masked_token_labels(原词标签)
    掩码采用了 80-10-10 策略：
    80%：my dog is hairy -> my dog is [MASK]。这是最主要的目标，让模型填空。
    10%：my dog is hairy -> my dog is apple。替换为随机词。这强迫模型观察上下文，纠正错误的词。
    10%：my dog is hairy -> my dog is hairy。保持不变。这非常重要，目的是为了让模型在微调（Fine-tuning）阶段（没有 [MASK] 标记时）也能产生好的向量表示，防止模型只针对 [MASK] 标记做预测。
    """

    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # 整词掩码意思是，如果我们掩码一个原始单词对应的全部词片段。
        # 当一个单词被拆分成词片段时，第一个标记没有前缀，而后续的标记都以前缀 “##” 开头。
        # 因此，每当我们看到 “##” 标记时，我们就会将它添加到前一组词索引中。

        # 请注意，整词掩码并不会改变任何训练代码——我们仍然独立地预测每个词片段，对整个词汇表进行softmax计算。
        """
        简单讲，一个长单词即使被拆成小token，会被打码为多个[MASK]；但预测是预测小token
        输入：["my", "hob", "##by", "is", "cod", "##ing"]
        输出 (cand_indices)：[[0], [1, 2], [3], [4, 5]]
        """
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)   # 打乱顺序，随机选择要 mask 的词

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # 如果添加一个全词掩模会超过预测的最大数值，那么可以跳过这个候选。
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        # 长度溢出检查和重复覆盖检查的代码
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        """遍历一个完整的单词（可能包含多个 token）, 按照8-1-1策略设置掩码"""
        for index in index_set:
            covered_indexes.add(index)

            # 80% 概率替换为[MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% 概率保留原词
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% 概率替换随机词
                else:
                    masked_token = choice(vocab_list)
            # 记录：MaskedLmInstance(位置索引, 原始真实标签)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    # 单词顺序被打乱，现在重新排序
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, bi_text=True):
    """
    功能: 将一个完整的文档（由多个句子组成）转换成多个 BERT 训练样本。
    此外，文档的抽样与其包含的句子数量成比例，这意味着每句话（而非每份文档）被作为 NextSentence 任务的虚假样本抽样的概率相等。

    输入：doc_database, doc_idx, max_seq_length(样本最大序列长度), short_seq_prob(生成较短序列的概率),
        masked_lm_prob(掩码概率), max_predictions_per_seq(单序列最大预测数), whole_word_mask(全词掩码开关), vocab_list(词表), bi_text(是否生成双文本实例)
    输出：instance字段: tokens/segment_ids(上下句分割掩码)/is_random_next(是否随机下一句)/masked_lm_positions(随机掩码位置)/masked_lm_labels(随机掩码源标签)
    """
    # 从单个文档生成多个训练样本(MLM+NSP)
    document = doc_database[doc_idx]
    # [CLS] A句 [SEP] B句 [SEP]
    max_num_tokens = max_seq_length - 3

    # 我们 通常 将整个句子填充至 “max_seq_length” 长度，那么短序列就会浪费很多计算资源。
    # 然而，我们 有时（即，short_seq_prob == 0.1 == 10% 的概率）想用更短的序列以最小化预训练与微调之间的不匹配。
    # 不过“target_seq_length”只是个粗糙的目标，而'max_seq_length'是个硬性限制。
    target_seq_length = max_num_tokens
    # 以一定概率（通常10%）生成随机长度的短序列
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # 我们不会把文档里的所有词缀串成一个长序列，然后随意选一个分点，因为那样下一个句子预测任务会太简单。
    # 相反，我们根据用户输入实际提供的内容，将输入分为“A”和“B”两个段。
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]   # 取出一个句子，已是tokens；document[0] = ["Hello", "world", "!"]
        current_chunk.append(segment)   # 以句子为单位！！
        current_length += len(segment)
        # 当攒够了目标长度，或者到了文档末尾，就开始处理这一块
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # “a_end”是“current_chunk”中进入 “前半句A” 的部分数量
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])   # 拼接 A 句的所有词

                tokens_b = []

                # 50% 概率随机下一句
                if bi_text and (len(current_chunk) == 1 or random() < 0.5):
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    """随即下一句，从别的文档里随机抽一些句子作为 tokens_b，较长的文档则更频繁地被采样（巧妙）"""
                    random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)

                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # 既然 B 是随机抽的，那么原本 current_chunk 里的后半部分就没用掉。
                    # 我们需要把指针 i 往回拨，让那些没用掉的句子在下一个训练样本里当 A 句。
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # 50% 概率真实下一句
                else:
                    # # 直接使用 current_chunk 里剩下的句子作为 tokens_b
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                if not tokens_a or len(tokens_a) == 0:
                    tokens_a = ["."]

                if not tokens_b or len(tokens_b) == 0:
                    tokens_b = ["."]

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # 如果 tokens_a + tokens_b 太长了，就从两头轮流删词，直到满足 max_num_tokens
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)   # 头尾随机逐个删除

                # 组装BERT输入并生成segment_id
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                # 生成 segment_ids (0代表A句，1代表B句)
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                """为掩码预测任务创建掩码，随机把 tokens 里的某些词变成 [MASK]"""
                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)

                # 单条训练样本的最终字典
                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels}

                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def create_training_file(docs, vocab_list, args, epoch_num, bi_text=True):
    """
    为一个epoch生成训练jsonl与metrics文件
    输入：docs, vocab_list(词表), args(运行参数), epoch_num, bi_text(是否生成双文本实例)
    输出：epoch_filename, metrics_filename
    """
    epoch_filename = args.output_dir / "epoch_{}.json".format(epoch_num)
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        for doc_idx in trange(len(docs), desc="Document"):
            """核心函数: 将一个完整的文档（由多个句子组成）转换成多个 BERT 训练样本。"""
            doc_instances = create_instances_from_document(
                docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list, bi_text=bi_text)
            # python字典转为json格式
            doc_instances = [json.dumps(instance) for instance in doc_instances]
            for instance in doc_instances:
                epoch_file.write(instance + '\n')
                num_instances += 1
    metrics_filename = args.output_dir / "epoch_{}_metrics.json".format(epoch_num)
    with metrics_filename.open('w') as metrics_file:
        metrics = {
            "num_training_examples": num_instances,
            "max_seq_len": args.max_seq_len
        }
        metrics_file.write(json.dumps(metrics))

    return epoch_filename, metrics_filename


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.0,
                        help="Probability of masking each token for the LM task")  # no [mask] symbol in corpus
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument('--data_url', type=str, default="")
    parser.add_argument('--one_seq', action='store_true')

    args = parser.parse_args()

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers while reducing memory")

    """初始化分词器与词表，basic + wordpiece"""
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())

    doc_num = 0
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        # 读取语料: 空行分隔文档，每行视为一句
        with args.train_corpus.open(encoding="utf-8") as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    docs.add_document(doc)
                    doc = []
                    doc_num += 1
                    if doc_num % 100 == 0:
                        logger.info('loaded {} docs!'.format(doc_num))
                else:
                    """调用核心函数: 将文档中每一行进行分词，basic + wordpiece"""
                    tokens = tokenizer.tokenize(line)
                    doc.append(tokens)
            if doc:
                docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added
        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")

        args.output_dir.mkdir(exist_ok=True)

        if args.num_workers > 1:
            # 多进程并行生成多个epoch的数据文件
            writer_workers = Pool(min(args.num_workers, args.epochs_to_generate))
            arguments = [(docs, vocab_list, args, idx) for idx in range(args.epochs_to_generate)]
            writer_workers.starmap(create_training_file, arguments)
        else:
            for epoch in trange(args.epochs_to_generate, desc="Epoch"):
                # one_seq=True时仅生成单句(无NSP)
                bi_text = True if not args.one_seq else False
                """调用核心函数: 生成训练样本"""
                epoch_file, metric_file = create_training_file(docs, vocab_list, args, epoch, bi_text=bi_text)

                # 是否在云端跑
                if oncloud:
                    logging.info(mox.file.list_directory(str(args.output_dir), recursive=True))
                    logging.info(mox.file.list_directory('.', recursive=True))
                    mox.file.copy_parallel(str(args.output_dir), args.data_url)
                    mox.file.copy_parallel('.', args.data_url)

                    os.remove(str(epoch_file))
                    os.remove(str(metric_file))


if __name__ == '__main__':
    main()
