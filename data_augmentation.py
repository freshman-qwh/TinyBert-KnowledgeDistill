# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd.
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
--pretrained_bert_model teacher_model --glove_embs 1_pre_data/glove.6B.100d.txt --glue_dir . --task_name SST-2
"""

import random
import sys
import os
import unicodedata
import re
import logging
import csv
import argparse


import torch
import numpy as np
from numpy.compat import unicode

from transformer import BertTokenizer, BertForMaskedLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

StopWordsList = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                 "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "'s", "'re"]


def strip_accents(text):
    """
    把输入字符串中的重音去除。

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    # 去除变音符号，保持ASCII范围
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError):
        # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


# valid string only includes al
def _is_valid(string):
    # 仅保留纯字母token，用于词级替换
    return True if not re.search('[^a-z]', string.lower()) else False


def _read_tsv(input_file, quotechar=None):
    """读取 TSV(tab separated value)文件并返回行列表"""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:    # 检查当前版本是否为 Python 2，但是3好像能跑？？？
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def prepare_embedding_retrieval(glove_file, vocab_size=100000):
    """读取 GloVe词向量并构建归一化矩阵, 用于相似词检索"""
    cnt = 0
    words = []
    embeddings = {}

    # GloVe通常有几百万词,只读前10万字以便快速检索, 排在前面的通常是高频词，涵盖了 99% 的常用语
    with open(glove_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            items = line.strip().split()
            words.append(items[0])  # 第一个元素是单词本身（如 "apple"）
            embeddings[items[0]] = [float(x) for x in items[1:]]    # 后面的是向量数值

            cnt += 1
            if cnt == vocab_size:   # 达到设定的词表上限（默认10万）就停止
                break

    vocab = {w: idx for idx, w in enumerate(words)}     # 单词 -> ID (如 {"apple": 0})
    ids_to_tokens = {idx: w for idx, w in enumerate(words)}     # ID -> 单词 (如 {0: "apple"})

    vector_dim = len(embeddings[ids_to_tokens[0]])      # 获取向量维度（如 100, 200 或 300）, ids_to_tokens[i]=words[i]
    emb_matrix = np.zeros((vocab_size, vector_dim))     # 初始化全 0 矩阵
    for word, v in embeddings.items():
        if word == '<unk>':
            continue
        emb_matrix[vocab[word], :] = v  # 将每个词的向量填入对应的矩阵行

    # 每行做归一化, 因为计算余弦相似度是S(A,B)=A·B/|A||B|, 归一化后|A|=|B|=1, S(A,B)=A·B
    # np.sum(emb_matrix ** 2, 1) 计算平方按行求和，** 0.5 开根号，得到每个向量的长度 d
    # [3, 4] -> [9, 16] -> 25 -> 5
    d = (np.sum(emb_matrix ** 2, 1) ** 0.5)
    # 归一化：将每个向量除以它的模长
    # emb_matrix.size=(100000,300)转置为(300,100000)才能/d, d.size=(100000,)
    emb_norm = (emb_matrix.T / d).T
    return emb_norm, vocab, ids_to_tokens


class DataAugmentor(object):
    def __init__(self, model, tokenizer, emb_norm, vocab, ids_to_tokens, M, N, p):
        """
        model: MaskedLM, tokenizer: BERT分词器
        emb_norm/vocab/ids_to_tokens: 用于近邻词检索
        M: 候选词数, N: 生成次数, p: 替换概率
        """
        self.model = model
        self.tokenizer = tokenizer
        self.emb_norm = emb_norm
        self.vocab = vocab
        self.ids_to_tokens = ids_to_tokens
        self.M = M
        self.N = N
        self.p = p

    def _word_distance(self, word):
        """基于词向量余弦相似度检索近邻词"""
        if word not in self.vocab.keys():
            return []
        word_idx = self.vocab[word]
        word_emb = self.emb_norm[word_idx]

        # 计算相似度, 屏蔽自身
        dist = np.dot(self.emb_norm, word_emb.T)    # 点乘与所有词向量的相似度，因为之前词向量矩阵已经归一化 S(A,B)=A·B
        dist[word_idx] = -np.Inf

        # 找TOP-M个词
        candidate_ids = np.argsort(-dist)[:self.M]  # np.argsort返回升序索引, -dist就是取最大值的索引
        candidate_words = [self.ids_to_tokens[idx] for idx in candidate_ids][:self.M]

        # 如果原词首字母大写（istitle），则替换词也首字母大写
        if word.istitle():
            candidate_words = [w.title() for w in candidate_words]
        return candidate_words

    def _masked_language_model(self, sent, word_pieces, mask_id):
        """
        基于 BERT 的上下文预测，引导 BERT 利用右侧的“原句信息”来修复左侧的 [MASK] 位置。
        输入：sent(原句), word_pieces(分词后tokens), mask_id(掩码位置索引)
        输出：word_candidates(预测候选,不带##)
        """
        # 总长度不超过512个token
        if mask_id >= 512:
            return []
        # 分词，制作上下文
        tokenized_text = self.tokenizer.tokenize(sent)
        tokenized_text = ['[CLS]'] + tokenized_text
        tokenized_len = len(tokenized_text)

        # word_pieces: ['[CLS]', 'I', 'like', '[MASK]']
        # tokenized_text[1:]: ['I', 'like', 'cat'] (去掉了CLS)
        # 总长度 len(tokenized_text) = 4 + 1 + 3 + 1 = 9
        tokenized_text = word_pieces + ['[SEP]'] + tokenized_text[1:] + ['[SEP]']

        if len(tokenized_text) > 512:
            tokenized_text = tokenized_text[:512]
            if tokenized_len >= 512:
                tokenized_len = 511

        # 计算上下句掩码
        # 0的个数 4+1=5， 1的个数 9-4-1=3
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * (tokenized_len + 1) + [1] * (len(tokenized_text) - tokenized_len - 1)

        # 将张量送入 BERT 模型
        tokens_tensor = torch.tensor([token_ids]).to(device)
        segments_tensor = torch.tensor([segments_ids]).to(device)

        self.model.to(device)

        predictions = self.model(tokens_tensor, segments_tensor)

        # 取top-M候选
        word_candidates = torch.argsort(predictions[0, mask_id], descending=True)[:self.M].tolist()
        word_candidates = self.tokenizer.convert_ids_to_tokens(word_candidates)
        # 删掉带 ## 的子词（WordPiece 的后缀），只保留完整的单词。
        return list(filter(lambda x: x.find("##"), word_candidates))

    def _word_augment(self, sentence, mask_token_idx, mask_token):
        """
        替换目标token，决定采用哪种策略（是问 BERT 还是问 GloVe）来寻找它的替代词。
        输入: sentence(原始的完整句子), mask_token_idx(BasicTokenizer基础分词后索引), mask_token(目标单词)
        输出: candidate_words(一个包含候选替换词的列表)
        BERT 的替换是动态的（受上下文影响）。例如 "bank" 在“银行”背景下会被替换为 "finance"，在“河岸”背景下会被替换为 "river"。
        GloVe 的替换是静态的。无论上下文，给出的都是最接近的近义词。
        """
        # ["playing", "football"] -> ["play", "##ing", "foot", "##ball"]
        word_pieces = self.tokenizer.tokenize(sentence)     # WordPiece 分词
        word_pieces = ['[CLS]'] + word_pieces
        tokenized_len = len(word_pieces)

        # 循环找到目标词在 BERT token 列表中的对应索引
        # 因为WordPiece 分词（如 playing -> play, ##ing）与普通分词位置不一致
        token_idx = -1
        for i in range(1, tokenized_len):
            # 如果这个 token 不含 "##"，说明它是一个新单词的开始
            if "##" not in word_pieces[i]:
                token_idx = token_idx + 1

                if token_idx < mask_token_idx:
                    word_piece_ids = []     # 还没搜到目标词，清空临时列表
                elif token_idx == mask_token_idx:
                    word_piece_ids = [i]    # 搜到了，记录当前索引
                else:
                    break
            # 如果包含 "##"，说明它是上一个单词的后缀，接在后面
            # 如果句子是 "The boy is playing" -> ['The', 'boy', 'is', 'playing'] ->
            # ['[CLS]', 'The', 'boy', 'is', 'play', '##ing'], 找基础分词索引3 "playing", word_piece_ids就是[4, 5]
            else:
                word_piece_ids.append(i)

        # 1. 目标词是单token, 把这个词换成 [MASK]。
        if len(word_piece_ids) == 1:
            word_pieces[word_piece_ids[0]] = '[MASK]'
            # 具体的构造mask和原句上下文，令BERT预测上下文
            candidate_words = self._masked_language_model(
                sentence, word_pieces, word_piece_ids[0])
        # 2. 目标词由多个 Tokens 组成, 问 GloVe 哪些词长得像
        elif len(word_piece_ids) > 1:
            candidate_words = self._word_distance(mask_token)   # glove相似度计算, 返回预选top-m列表
        else:
            logger.info("invalid input sentence!")

        # 3. 如果都没找到，就用原词，不改变
        if len(candidate_words)==0:
            candidate_words.append(mask_token)

        return candidate_words

    def augment(self, sent):
        """
        对单句生成N条增强样本
        输出: [原句, 增强句1, ...]
        """
        candidate_sents = [sent]
        # 基础分词
        tokens = self.tokenizer.basic_tokenizer.tokenize(sent)
        candidate_words = {}
        for (idx, word) in enumerate(tokens):
            # 过滤非字母、大写、停用词
            if _is_valid(word) and word.lower() not in StopWordsList:
                """重点，调用增强，针对某个单词生成候选库；可用MLM增强或GloVe增强"""
                candidate_words[idx] = self._word_augment(sent, idx, word)
        # logger.info(candidate_words)
        cnt = 0
        while cnt < self.N:
            new_sent = list(tokens)
            for idx in candidate_words.keys():
                # 随机去一个候选词
                candidate_word = random.choice(candidate_words[idx])
                # 小概率替换为候选词，通过较小的概率p，确保每生成的一条新句子只在原句的基础上微调了 1~2 个单词
                x = random.random()
                if x < self.p:
                    new_sent[idx] = candidate_word

            # 去重与保存
            if " ".join(new_sent) not in candidate_sents:
                candidate_sents.append(' '.join(new_sent))
            cnt += 1

        return candidate_sents


class AugmentProcessor(object):
    def __init__(self, augmentor, glue_dir, task_name):
        # glue_dir: GLUE数据目录, task_name: 子任务名
        self.augmentor = augmentor
        self.glue_dir = glue_dir
        self.task_name = task_name
        self.augment_ids = {'MRPC': [3, 4], 'MNLI': [8, 9], 'CoLA': [3], 'SST-2': [0],
                            'STS-B': [7, 8], 'QQP': [3, 4], 'QNLI': [1, 2], 'RTE': [1, 2]}

        self.filter_flags = { 'MRPC': True, 'MNLI': True, 'CoLA': False, 'SST-2': True,
                              'STS-B': True, 'QQP': True, 'QNLI': True, 'RTE': True}

        assert self.task_name in self.augment_ids

    def read_augment_write(self):
        # 读取train.tsv, 对指定列进行增强, 输出train_aug.tsv
        task_dir = os.path.join(self.glue_dir, self.task_name)
        train_samples = _read_tsv(os.path.join(task_dir, "train.tsv"))
        output_filename = os.path.join(task_dir, "train_aug.tsv")
        # 加载任务参数
        augment_ids_ = self.augment_ids[self.task_name]
        filter_flag = self.filter_flags[self.task_name]

        with open(output_filename, 'w', newline='', encoding="utf-8") as f:
            aug_lines = 0
            # 使用 csv 模块创建一个写入器，delimiter="\t" 表示生成的格式是 TSV（BERT 常用格式）
            writer = csv.writer(f, delimiter="\t")
            for (i, line) in enumerate(train_samples):
                # 如果是第一行且 filter_flag 为 True，通常意味着这一行是表头（如 "sentence\tlabel"）
                if i == 0 and filter_flag:
                    writer.writerow(line)   # 标题行直接写入，不进行增强
                    continue

                for augment_id in augment_ids_:
                    sent = line[augment_id]     # 提取需要增强的那一列（例如句子 A）
                    """为每条句子增强生成至多N条语句"""
                    augmented_sents = self.augmentor.augment(sent)  # 生成 N 条句子
                    for augment_sent in augmented_sents:
                        # 将当前行对应的列替换为增强后的句子
                        line[augment_id] = augment_sent
                        # 将整行数据（包含其他的列，如标签 label）写入文件
                        writer.writerow(line)
                        aug_lines += 1

                if (i+1) % 10 == 0:
                    logger.info("Having been processing {} examples".format(str(i+1)))

                if i >= 200:
                    logger.info("已经生成约 {} 条数据，学习版调试够用了，中止增强".format(str(aug_lines)))
                    break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_bert_model", default=None, type=str, required=True,
                        help="Downloaded pretrained model (bert-base-cased/uncased) is under this folder")
    parser.add_argument("--glove_embs", default=None, type=str, required=True,
                        help="Glove word embeddings file")
    parser.add_argument("--glue_dir", default=None, type=str, required=True,
                        help="GLUE data dir")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="Task(eg. CoLA, SST-2) that we want to do data augmentation for its train set")
    parser.add_argument("--N", default=30, type=int,
                        help="How many times is the corpus expanded?")
    parser.add_argument("--M", default=15, type=int,
                        help="Choose from M most-likely words in the corresponding position")
    parser.add_argument("--p", default=0.4, type=float,
                        help="Threshold probability p to replace current word")

    args = parser.parse_args()
    # logger.info(args)

    # 任务对应的默认增强倍数
    default_params = {
        "CoLA": {"N": 30},
        "MNLI": {"N": 10},
        "MRPC": {"N": 30},
        "SST-2": {"N": 20},
        "STS-b": {"N": 30},
        "QQP": {"N": 10},
        "QNLI": {"N": 20},
        "RTE": {"N": 30}
    }

    if args.task_name in default_params:
        args.N = default_params[args.task_name]["N"]

    # 准备数据增强器: MaskedLM + GloVe近邻
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)
    model = BertForMaskedLM.from_pretrained(args.pretrained_bert_model)
    model.eval()

    """读取 GloVe词向量并构建归一化矩阵, 用于相似词检索"""
    emb_norm, vocab, ids_to_tokens = prepare_embedding_retrieval(args.glove_embs)
    # 创建增强示例
    data_augmentor = DataAugmentor(model, tokenizer, emb_norm, vocab, ids_to_tokens, args.M, args.N, args.p)

    """执行数据增强并写出train_aug.tsv"""
    processor = AugmentProcessor(data_augmentor, args.glue_dir, args.task_name)
    processor.read_augment_write()


if __name__ == "__main__":
    main()
