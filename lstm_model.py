# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import time
from utils import get_logger
from data_helper import batch_iter
import numpy as np
from utils import focal_loss


class BiLSTM_Model(object):
    def __init__(self, args, tag2label, model_w2v, paths, config):
        self.batch_size = args['batch_size']
        self.epoch_num = args['epoch']
        self.hidden_dim = args['hidden_dim']
        self.embedding_dim = args['embedding_dim']
        self.dropout_keep_prob = args['dropout']
        self.lr_ = args['lr']
        # 学习率自动衰减
        self.global_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(self.lr_, self.global_step, decay_steps=4,
                                             decay_rate=0.98, staircase=False)
        # 词向量矩阵，添加一行全0向量，对应没出现过的词<UNK>
        self.embeddings = np.r_[model_w2v.wv.vectors, [[0] * self.embedding_dim]]
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.shuffle = args['shuffle']
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        """
        构建TensorFlow网络结构
        可以添加attention层去优化
        :return:
        """
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.accuracy_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        """
        使用tf的占位符，初始化3个属性
        :return:
        """
        self.word2ids = tf.placeholder(tf.int32, shape=[None, None], name="word2ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    def lookup_layer_op(self):
        """
        在预训练的词向量矩阵中寻找max_len个词向量，一个word_id对应一个词向量，id数即为词向量矩阵的第几行
        预训练词向量矩阵是一个可优化的点
        """
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word2ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            # 前向lstm
            cell_fw = LSTMCell(self.hidden_dim)
            # 反向lstm
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                dtype=tf.float32)

            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)

            # 将词向量取平均，维度变化：[batch_size, max_len, hidden_dim] -> [batch_size, hidden_dim]
            # 即将max_len个hidden_dim维的词向量合并为一个
            output = tf.reduce_sum(output, axis=1)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            pred = tf.matmul(output, W) + b
            self.logits = tf.nn.dropout(pred, self.dropout_pl)

    def loss_op(self):
        """
        损失函数，可尝试替换其他损失函数去优化，如focal loss
        """
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)
        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", self.loss)

    def accuracy_op(self):
        self.prediction = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
        correct_prediction = tf.equal(self.prediction, tf.cast(tf.argmax(self.labels, axis=1), tf.int32))
        self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="acc")

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optim.minimize(self.loss)

    def init_op(self):
        """变量初始化"""
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """
        打印概览
        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """
        训练入口
        :param train:训练集
        :param dev: 验证集
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, epoch, saver)

    def test(self, test):
        """
        测试入口
        :param test:测试集
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            self.dev_one_epoch(sess=sess, dev=test)

    def demo(self, demo_sent):
        """
        测试一条文本，返回值可根据需求更改
        """
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            saver.restore(sess, self.model_path)
            y_pred = self.predict_one(sess, demo_sent)[0]
            label = self.tag2label[str(y_pred)]
            return label

    def run_one_epoch(self, sess, train, dev, epoch, saver):
        """
        训练1个opoch
        :param sess:session
        :param train:训练集
        :param dev:验证集
        :param epoch:当前epoch
        :param saver:tf模型保存对象
        :return:
        """
        num_batches = (len(train[0]) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_iter(train, self.batch_size)
        avg_loss = 0
        avg_acc = 0
        for step, (seqs, labels) in enumerate(batches):
            step_num = epoch * num_batches + step + 1
            feed_dict = self.get_feed_dict(seqs, labels, self.dropout_keep_prob)
            _, loss_train, acc_train, summary, step_num_ = sess.run(
                [self.train_op, self.loss, self.acc, self.merged, self.global_step],
                feed_dict=feed_dict)
            avg_loss += loss_train / num_batches
            avg_acc += acc_train / num_batches
            self.file_writer.add_summary(summary, step_num)
        self.logger.info(
            '{} epoch {}, loss: {:.4}, acc: {:.4}'.format(start_time, epoch + 1, avg_loss, avg_acc))
        saver.save(sess, self.model_path, global_step=epoch + 1)

        self.logger.info('===========validation / test===========')
        self.dev_one_epoch(sess, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, dropout=None):
        """
        获取tf网络模型需要“喂”的数据
        :param seqs: 文本
        :param labels:
        :param dropout:
        :return: feed_dict
        """
        feed_dict = {self.word2ids: seqs}
        if labels is not None:
            feed_dict[self.labels] = labels
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict

    def dev_one_epoch(self, sess, dev, epoch=None):
        """
        :param sess:
        :param dev:
        :param epoch:
        :return:
        """
        num_batches = (len(dev[0]) + self.batch_size - 1) // self.batch_size
        avg_acc = 0
        for seqs, labels in batch_iter(dev, self.batch_size):
            feed_dict = self.get_feed_dict(seqs, labels, 1.0)
            acc = sess.run(self.acc, feed_dict=feed_dict)
            avg_acc += acc / num_batches
        if epoch is not None:
            self.logger.info('epoch {}, dev acc : {:.4}'.format(epoch + 1, avg_acc))
        else:
            self.logger.info('Test acc : {:.4}'.format(avg_acc))

    def predict_one(self, sess, seqs):
        """
        预测一条
        :param sess:
        :param seqs:
        :return y_pred: 预测标签对应的id
        """
        seqs = np.array([seqs])
        feed_dict = self.get_feed_dict(seqs=seqs, dropout=1.0)
        y_pred = sess.run(self.prediction, feed_dict=feed_dict)

        return y_pred
