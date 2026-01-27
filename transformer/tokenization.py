# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open


logger = logging.getLogger(__name__)

# 词汇表下载地址
PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}

PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}

VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file):
    """
    将词汇文件加载到词典中。
    输出：{"token":index}
    """
    # 读取vocab.txt: token -> id
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """运行基本的空白清理和文本分割。"""
    text = text.strip()     # 清除两端
    if not text:
        return []
    tokens = text.split()   # 空格分词
    return tokens


class BertTokenizer(object):

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True, basic_only=False,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """构建一个BertTokenizer
        运行端到端分词：标点拆分+单词片段的令牌解析器。

        Args:
          vocab_file: 路径指向每行单词的词汇文件
          do_lower_case: 是否将输入小写
                         仅在 do_wordpiece_only=False 时才有影响
          do_basic_tokenize: 是否在单词前做基础的分词。
          max_len: 一个人为的最大长度，用于截断分词序列：有效最大长度总是其中最小值（如指定）以及底层BERT模型的序列长度。
          never_split: token列表，这些token在token化过程中永远不会被拆分。
                         仅在 do_wordpiece_only=False 时才有影响
        """
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        """将词汇文件加载到词典中。tokens->id"""
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        # 基本的分词处理
        if do_basic_tokenize:
          self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                never_split=never_split)
        # 贪心最长匹配优先的wordpiece token分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
        self.basic_only = basic_only

    def tokenize(self, text):
        """输出WordPiece序列(list[str])"""
        split_tokens = []
        if self.do_basic_tokenize:
            """
            运行基本的分词处理（标点拆分、下写等），多语言模型需要。
            以及WordPiece分词，将一个单词拆成多个token。
            """
            for token in self.basic_tokenizer.tokenize(text):
                if self.basic_only:
                    split_tokens.append(token)
                # 继续分词，单词拆成更细的token
                else:
                    for sub_token in self.wordpiece_tokenizer.tokenize(token):
                        split_tokens.append(sub_token)
        # 直接拆分成token
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """利用词汇将 tokens -> IDs, 未知token映射为[UNK]。"""
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab['[UNK]']))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """利用词汇将 IDs-> tokens。"""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def save_vocabulary(self, vocab_path):
        """保存tokenizer vocabulary"""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return vocab_file

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        从预训练的BertModel模型文件中，实例化一个预训练的Tokenizer。
        如有需要，下载并缓存预训练的模型文件。
        """

        # assert PRETRAINED_VOCAB_ARCHIVE_MAP 中的 pretrained_model_name_or_path
        resolved_vocab_file = os.path.join(pretrained_model_name_or_path, 'vocab.txt')

        max_len = 512
        kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # 实例化令牌器
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)

        return tokenizer


class BasicTokenizer(object):

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """
        运行基本的分词处理（标点拆分、下写等）。
        清洗文本 -> 中文字符分隔 -> 标点拆分 -> 空白规整
        Args:
          do_lower_case: 是否要把输入写小写。
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """清洗文本 -> 中文字符分隔 -> 标点拆分 -> 空白规整"""
        text = self._clean_text(text)
        # 该功能于2018年11月1日新增，适用于多语言和中文模型。
        # 这也适用于英语模型，但这无关紧要，因为英语模型没有用中文数据训练，通常也没有中文数据（除非词汇中包含汉字，因为维基百科英文维基百科中确实有一些中文单词）。

        # 在中日韩字符两侧加空格，并分词
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        # 去除重音符号(拉丁字符), 将标点符号单独切分
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """去除重音符号(拉丁字符)."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """将标点符号单独切分"""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """在中日韩字符两侧加空格，便于分词"""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """检查是否是中日汉字的码点。"""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """去除控制字符/非法字符，统一空白"""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """运行 WordPiece token化。"""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """将一段文本分成单词片段。

        该算法使用贪婪的最长匹配优先算法，利用给定词汇进行分词。

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: 单个token或空白分隔token。这应该已经通过“BasicTokenizer”传递过了。

        Returns:
          一份单词标记列表。
        """

        # 使用贪心最长匹配进行WordPiece切分
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 单词过长转为 unknow
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """空白字符判断"""
    # \t、\n和\r技术上是扭曲字符，但我们把它们当作空白，因为它们通常被当作空白。
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """控制字符判断"""
    # 这些技术上是控制字符，但我们算作空白字符。
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """标点符号判断"""
    cp = ord(char)
    # 我们把所有非字母/数字的ASCII都当作标点符号处理。
    # 像“^”、“$”和“'”这样的字符不在Unicode中
    # 标点课，但我们还是把它们当作标点来对待，以保持一致性。
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
