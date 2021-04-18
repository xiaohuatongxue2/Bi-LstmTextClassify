# -*- coding: utf-8 -*-
import logging
import tensorflow as tf


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def focal_loss(logits, labels, alpha, batch_size, epsilon=1.e-7, gamma=2.0):
    """
    样本不均衡考虑的损失函数
    """
    # (Class ,1)
    alpha = tf.constant(alpha, dtype=tf.float32)

    labels = tf.cast(labels, dtype=tf.int32)
    logits = tf.cast(logits, tf.float32)
    # (N,Class) > N*Class
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])
    # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
    labels_shift = tf.range(0, batch_size) * logits.shape[1] + labels
    # labels_shift = tf.range(0, batch_size*32) * logits.shape[1] + labels
    # (N*Class,) > (N,)
    prob = tf.gather(softmax, labels_shift)
    # 预防预测概率值为0的情况  ; (N,)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    # (Class ,1) > (N,)
    alpha_choice = tf.gather(alpha, labels)
    # (N,) > (N,)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.multiply(alpha_choice, weight)
    # (N,) > 1
    loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
    return loss
