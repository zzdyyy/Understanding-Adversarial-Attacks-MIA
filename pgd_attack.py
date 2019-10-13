"""
https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py

Towards Deep Learning Models Resistant to Adversarial Attacks 
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu 
https://arxiv.org/abs/1706.06083.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
# from util import lid

class LinfPGDAttack:
    def __init__(self, model, epsilon, eps_iter, nb_iter, kappa=0, random_start=False,
                 loss_func='xent', clip_min=0.0, clip_max=1.0):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.loss_func = loss_func
        self.epsilon = epsilon
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.kappa = kappa
        self.rand = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.x_input = self.model.layers[0].input
        self.logits = self.model.layers[-1].output
        # self.logits = self.model.get_layer("logits").output
        self.y_pred = tf.nn.softmax(self.logits)
        self.y_true = tf.placeholder(tf.float32, shape=self.y_pred.get_shape().as_list())

        if loss_func == 'xent':
            self.loss = -tf.reduce_sum(self.y_true * tf.log(self.y_pred + 1e-12), axis=-1)
        elif loss_func == 'cw':
            correct_logit = tf.reduce_sum(self.y_true * self.logits, axis=-1)
            wrong_logit = tf.reduce_max((1 - self.y_true) * self.logits, axis=-1)
            self.loss = -tf.nn.relu(correct_logit - wrong_logit + kappa)
        # elif loss_func == 'cw-lid':
        #     lids = lid(self.logits, k=20)
        #     self.loss = self.xent + 0.1 * lids
        elif loss_func == 'trades':
            self.p_natural = tf.placeholder(tf.float32, shape=self.y_pred.get_shape().as_list())
            self.p_natural = tf.clip_by_value(self.p_natural, 1e-7, 1.0)
            self.y_pred = tf.clip_by_value(self.y_pred, 1e-7, 1.0)
            self.loss = tf.reduce_sum(self.p_natural * tf.log(self.p_natural / self.y_pred + 1e-12), axis=1)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss = -tf.reduce_sum(self.y_true * tf.log(self.y_pred), axis=-1)

        self.grad = tf.gradients(self.loss, self.x_input)[0]

    def perturb(self, sess, x_nat, y, batch_size):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            if self.loss_func == 'trades':
                x = x_nat + 0.001 * np.random.normal(loc=0.0, scale=1.0, size=x_nat.shape)
            else:
                x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, self.clip_min, self.clip_max) # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        nb_batch = len(x) // batch_size
        # check if need one more batch
        if nb_batch * batch_size < len(x):
            nb_batch += 1
        for i in range(nb_batch):
            start = i * batch_size
            end = (i + 1) * batch_size
            end = np.minimum(end, len(x))
            batch_x = x[start:end]
            batch_y = y[start:end]

            p_nat, logits_nat = sess.run([self.y_pred, self.logits],
                                         feed_dict={self.x_input: x_nat[start:end]})

            for j in range(self.nb_iter):
                if self.loss_func == 'trades':
                    loss, grad = sess.run([self.loss, self.grad],
                                          feed_dict={self.x_input: batch_x,
                                                     self.p_natural: p_nat})
                    grad = np.nan_to_num(grad)
                    batch_x += self.eps_iter * np.sign(grad)
                else:
                    loss, grad = sess.run([self.loss, self.grad],
                                          feed_dict={self.x_input: batch_x,
                                                     self.y_true: batch_y})
                    batch_x += self.eps_iter * np.sign(grad)
                batch_x = np.clip(batch_x, x_nat[start:end] - self.epsilon, x_nat[start:end] + self.epsilon)
                batch_x = np.clip(batch_x, self.clip_min, self.clip_max)  # ensure valid pixel range
            x[start:end] = batch_x[:]
        return x

