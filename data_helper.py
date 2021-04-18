# -*- coding: utf-8 -*-
import json
import pandas as pd
import jieba
import os
import re
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import joblib
from sklearn.model_selection import train_test_split
import random


CUR_PATH = os.path.dirname(__file__)


def read_data_from_table(file_path):
    """
    读取数据
    """
    if file_path.endswith('.csv'):
        data_df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data_df = pd.read_excel(file_path)
    else:
        raise Exception('Can not read {}, please check you file name.'.format(file_path))
    return data_df


def split_train_test_set(file_path, train_path, test_path):
    data_set = read_data_from_table(file_path)
    train, test = train_test_split(data_set, test_size=0.3, random_state=666)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


def label_encoder(x, cat2id):
    """
    标签转数字
    """
    if str(x).strip() in cat2id:
        return cat2id[str(x).strip()]
    else:
        raise Exception('Not Found {} in cat2id'.format(str(x)))


def label_decoder(x, id2cat):
    """
    数字转标签
    """
    if str(x) in id2cat:
        return id2cat[str(x)]
    else:
        raise Exception('Not Found {} in id2cat.'.format(str(x)))


def build_label2tag(label2tag_path):
    """
    构建标签转数字json
    :parma label2tag_path: 标签转数字json文件保存路径
    """
    categories = ['positive', 'neutral', 'negative']
    label2tag = {}
    tag2label = {}
    label_json = {}
    for index, category in enumerate(categories):
        label2tag[category] = index
        tag2label[index] = category
    label_json['label2tag'] = label2tag
    label_json['tag2label'] = tag2label
    with open(label2tag_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(label_json, ensure_ascii=False))


def load_label2tag(label2tag_path):
    """
    加载标签转数字json
    :parma label2tag_path: 标签转数字json文件保存路径
    """
    with open(label2tag_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_stop_words(file_path):
    """
    加载停用词
    """
    return [line.strip() for line in open(file_path, 'r', encoding='utf-8')]


def data_process(text, stopwords):
    """
    对文本进行预处理：清除无用字符，去除停用词，分词
    :param text: 评论文本，
    :param stopwords：停用词
    :return 预处理后的文本
    """
    text = str(text).replace('\n', ' ').strip()
    text = re.sub('\s+', ' ', text)
    text = re.sub('[,，。 /:：]', ',', text)
    text = re.sub(',+', ',', text)
    word_list = list(jieba.cut(text))
    word_list = [x for x in word_list if x not in stopwords]
    word_list = list(set(word_list))
    return ' '.join(word_list)


def build_corpus(corpus_list, corpus_path):
    """
    构建语料，用来训练word2vec
    :param corpus_list: 预处理后的评论数据
    :param corpus_path: 语料保存路径
    """
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for sen in corpus_list:
            f.write(sen)
            f.write('\n')


def built_word2vec_model(corpus_path, w2v_path):
    """
    构建word2vec词向量模型
    :parma corpus_path: 预料文件
    :w2v_path: 保存的word2vec模型的路径
    """
    model = Word2Vec(LineSentence(corpus_path), size=128, window=5, min_count=3)
    model.save(w2v_path)


def word_embedding(text, word2id, max_len=100):
    """
    将文本转成word2vec对应的id向量
    """
    vector = []
    for word in text.split():
        if str(word) in word2id:
            vector.append(word2id[str(word)])
        else:
            for char in str(word):
                if str(char) in word2id:
                    vector.append(word2id[str(char)])
    vector = vector[: max_len] + [0] * (max_len - len(vector))
    return vector


def build_word2id(w2v_model, word2id_path):
    """
    构建单词表
    """
    word2id = {}
    for index, word in enumerate(w2v_model.wv.index2word):
        word2id[word] = index
    word2id['<UNK>'] = len(word2id) + 1
    with open(word2id_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(word2id))


def label_padding(x, label_len):
    pad = [0] * label_len
    pad[x] = 1
    return pad


def get_pre_model(label2tag_path, corpus_path, w2v_path, word2id_path):
    if not os.path.exists(label2tag_path):
        build_label2tag(label2tag_path)
    if not os.path.exists(w2v_path):
        built_word2vec_model(corpus_path, w2v_path)
    model_w2v = joblib.load(w2v_path)
    if not os.path.exists(word2id_path):
        build_word2id(model_w2v, word2id_path)
    label_data = load_label2tag(label2tag_path)
    with open(word2id_path, 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    return label_data, word2id, model_w2v


def generate_one_vec(text, word2id, stopwords):
    text = data_process(text, stopwords)
    embedding = word_embedding(text, word2id)
    return embedding


def build_data_set(data_path, label2tag_path, word2id_path, stopwords, corpus_path, w2v_path):
    """
    构建数据集
    :return data_set: 训练集
    """
    data_set = read_data_from_table(data_path)
    data_set['评论内容'] = data_set['评论内容'].apply(lambda x: data_process(x, stopwords))
    if not os.path.exists(corpus_path):
        build_corpus(data_set['评论内容'], corpus_path)

    label_json, word2id, model_w2v = get_pre_model(label2tag_path, corpus_path, w2v_path, word2id_path)
    data_set['标签'] = data_set['标签'].apply(lambda x: label_encoder(x, label_json['label2tag']))

    x_list = []
    y_list = []
    for sen, label in zip(data_set['评论内容'], data_set['标签']):
        x_list.append(word_embedding(sen, word2id))
        y_list.append(label_padding(label, len(label_json['label2tag'])))
    # 过采样

    return (x_list, y_list), label_json, model_w2v


def batch_iter(data, batch_size, shuffle=True):
    """
    批次数据生成器
    """
    x, y = data
    data_size = len(x)
    num_batches_per_epoch = int((data_size - 1) / batch_size)
    if shuffle:
        shuffle_indices = list(range(data_size))
        random.shuffle(shuffle_indices)
        shuffled_x = []
        shuffled_y = []
        for shuffle_indice in shuffle_indices:
            shuffled_x.append(x[shuffle_indice])
            shuffled_y.append(y[shuffle_indice])
    else:
        shuffled_x = x
        shuffled_y = y
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_size)
        return_x = shuffled_x[start_index:end_index]
        return_y = shuffled_y[start_index:end_index]
        yield return_x, return_y


if __name__ == '__main__':
    source_path = 'data/source/all_mfw.csv'
    train_path = 'data/source/trainSet.csv'
    test_path = 'data/source/testSet.csv'
    # 划分数据集
    split_train_test_set(source_path, train_path, test_path)
