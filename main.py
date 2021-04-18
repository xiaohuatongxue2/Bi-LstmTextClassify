# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import time

from utils import get_logger
from data_helper import build_data_set, generate_one_vec, get_pre_model
from lstm_model import BiLSTM_Model
import multiprocessing
from imblearn.over_sampling import SMOTE
import numpy as np

cur_path = os.path.dirname(__file__)
lock = multiprocessing.Lock()


def get_config(data_path, run_type='train'):
    """
    设置参数
    :return: args, label_json, word2id, paths, config,train_data,test_data
            超参数、标签转数字，词转数字，路径词典，环境配置，训练数据，测试数据，
    """
    # 训练显卡配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    # 配置超参数
    args = {}
    args = get_hyper_parameter(args)

    # 输出路径设置
    paths = {}
    paths = set_path(paths, args, run_type)

    stopwords_path = os.path.join(paths['source_path'], 'hit_stopwords.txt')
    stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
    label2tag_path = os.path.join(paths['output_path'], 'label2tag.json')
    corpus_path = os.path.join(paths['output_path'], 'corpus.txt')
    w2v_path = os.path.join(paths['output_path'], 'model_w2v.pkl')
    word2id_path = os.path.join(paths['output_path'], 'word2id.json')

    # 获取数据集
    if run_type == 'demo':
        label_json, word2id, model_w2v = get_pre_model(label2tag_path, corpus_path, w2v_path, word2id_path)
        return args, paths, label_json, word2id, model_w2v, config
    if run_type == 'train':
        data_set, label_json, model_w2v = build_data_set(data_path, label2tag_path, word2id_path, stopwords, corpus_path, w2v_path)

        # 训练集过采样
        # x_list, y_list = data_set
        # X = np.array(x_list)
        # y = np.array(y_list)
        # smo = SMOTE(random_state=42)
        # x_list, y_list = smo.fit_sample(X, y)
        # data_set = (x_list, y_list)
    elif run_type == 'test':
        data_set, label_json, model_w2v = build_data_set(data_path, label2tag_path, word2id_path, stopwords, corpus_path, w2v_path)
    else:
        raise Exception('Run type is error')
    return data_set, args, paths, label_json, model_w2v, config


def get_hyper_parameter(args):
    """
    配置超参数
    :param args:
    :return:
    """
    args['batch_size'] = 32
    args['epoch'] = 8
    args['hidden_dim'] = 128
    args['optimizer'] = 'Adam'
    args['lr'] = 0.01
    args['dropout'] = 0.5
    args['embedding_dim'] = 128
    args['classes_num'] = 3
    args['shuffle'] = True
    # 测试和demo用的版本号
    args['demo_model'] = '03211055'

    return args


def set_path(paths, args, run_type):
    """
    配置路径
    :param paths: 路径词典
    :param args: 参数列表
    :param run_type:
    :return: 配置好的path
    """
    timestamp = time.strftime('%m%d%H%M', time.localtime(time.time())) if run_type == 'train' else args['demo_model']
    data_path = os.path.join(cur_path, 'data')
    paths['output_path'] = os.path.join(data_path, 'output')
    paths['source_path'] = os.path.join(data_path, 'source')
    output_path = os.path.join(paths['output_path'], timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    if run_type == 'train':
        # 用于存储模型文件
        paths['model_path'] = ckpt_prefix
    else:
        # 用于读取模型文件
        paths['model_path'] = model_path
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path
    get_logger(log_path).info(str(args))

    return paths


def train(train_set, valid_set, args, paths, label_json, model_w2v, config):
    model = BiLSTM_Model(args, label_json['tag2label'], model_w2v, paths, config)
    model.build_graph()
    model.train(train=train_set, dev=valid_set)


def test(test_set, args, paths, label_json, model_w2v, config):
    ckpt_file = tf.train.latest_checkpoint(paths['model_path'])
    paths['model_path'] = ckpt_file
    model = BiLSTM_Model(args, label_json['tag2label'], model_w2v, paths, config)
    model.build_graph()
    print("test data: {}".format(len(test_set[0])))
    model.test(test_set)


def demo(args, label_json, word2id, model_w2v, paths, config):
    """
    预测单个文本
    """
    ckpt_file = tf.train.latest_checkpoint(paths['model_path'])
    paths['model_path'] = ckpt_file
    stopwords_path = os.path.join(paths['source_path'], 'hit_stopwords.txt')
    stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
    model = BiLSTM_Model(args, label_json['tag2label'], model_w2v, paths, config)
    model.build_graph()

    while True:
        text = input('请输入文本: \n')
        embedding = generate_one_vec(text, word2id, stopwords)
        label = model.demo(embedding)
        info = {
            '情感倾向': label,
        }
        print(info)


if __name__ == '__main__':
    train_path = 'data/source/trainSet.csv'
    test_path = 'data/source/testSet.csv'

    # 训练过程
    # train_set, args, paths, label_json, model_w2v, config = get_config(train_path, run_type='train')
    # valid_set, _, _, _, _, _ = get_config(test_path)
    # train(train_set, valid_set, args, paths, label_json, model_w2v, config)

    # 测试过程
    test_set, args, paths, label_json, model_w2v, config = get_config(test_path, run_type='test')
    test(test_set, args, paths, label_json, model_w2v, config)

    # demo
    # args, paths, label_json, word2id, model_w2v, config = get_config(data_path=None, run_type='demo')
    # demo(args, label_json, word2id, model_w2v, paths, config)
