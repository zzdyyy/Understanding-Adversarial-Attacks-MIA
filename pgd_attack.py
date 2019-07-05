"""
https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py

Towards Deep Learning Models Resistant to Adversarial Attacks 
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu 
https://arxiv.org/abs/1706.06083.

https://github.com/anishathalye/obfuscated-gradients/blob/master/lid/lid.ipynb
"""
import tensorflow as tf
import numpy as np
import keras.backend as K


class LinfPGDAttack:
    def __init__(self, model, tol, num_steps, step_size, random_start):
        self.model = model
        self.tol = tol
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start

        self.xs = tf.Variable(np.zeros((10, 32, 32, 3), dtype=np.float32),
                              name='modifier')
        self.orig_xs = tf.placeholder(tf.float32, [None, 32, 32, 3])

        self.ys = tf.placeholder(tf.int32, [None])

        self.epsilon = 8.0 / 255

        delta = tf.clip_by_value(self.xs, 0, 255) - self.orig_xs
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs = tf.assign(self.xs, self.orig_xs + delta)

        self.logits = logits = model(self.xs)

        label_mask = tf.one_hot(self.ys, 10)
        correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
        wrong_logit = tf.reduce_max((1 - label_mask) * logits - 1e4 * label_mask, axis=1)

        self.loss = (correct_logit - wrong_logit)

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(step_size * 1)

        grad, var = optimizer.compute_gradients(self.loss, [self.xs])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad), var)])

        end_vars = tf.global_variables()
        self.new_vars = [x for x in end_vars if x.name not in start_vars]

    def perturb(self, x, y, sess):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x, K.learning_phase(): 0})

        for i in range(self.num_steps):
            sess.run(self.train, feed_dict={self.ys: y, K.learning_phase(): 0})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x, K.learning_phase(): 0})

        return sess.run(self.xs)

