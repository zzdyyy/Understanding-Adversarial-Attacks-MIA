#### This is just a copy and paste from https://github.com/carlini/nn_robust_attacks.
## This include the CarliniL2, CarliniLi attack, and the adapted CarliniLID attack to
## break the LID-based subspace detection in ICLR2018 paper:
## Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality.
# https://arxiv.org/abs/1801.02613
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>. for CarliniL2 and CarliniLi
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from cleverhans.utils import other_classes
import keras.backend as K

from util import lid_adv_term

# settings for C&W L2 attack
L2_BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
L2_MAX_ITERATIONS = 1000    # number of iterations to perform gradient descent
L2_ABORT_EARLY = True       # if we stop improving, abort gradient descent early
L2_LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
L2_TARGETED = True          # should we target one specific class? or just be wrong?
L2_CONFIDENCE = 0           # how strong the adversarial example should be
L2_INITIAL_CONST = 1e-3    # the initial constant c to pick as a first guess

class CarliniL2:
    def __init__(self, sess, model, image_size, num_channels, num_labels, batch_size=100,
                 confidence=L2_CONFIDENCE, targeted=L2_TARGETED, learning_rate=L2_LEARNING_RATE,
                 binary_search_steps=L2_BINARY_SEARCH_STEPS, max_iterations=L2_MAX_ITERATIONS,
                 abort_early=L2_ABORT_EARLY,
                 initial_const=L2_INITIAL_CONST):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """
        self.model = model
        self.sess = sess
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        shape = (self.batch_size, self.image_size, self.image_size, self.num_channels)

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))
        self.max_mod = tf.reduce_max(modifier)

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((self.batch_size, self.num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (self.batch_size, self.num_labels))
        self.assign_const = tf.placeholder(tf.float32, [self.batch_size])

        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        self.newimg = tf.tanh(modifier + self.timg) / 2

        # prediction BEFORE-SOFTMAX of the model
        self.output = self.model(self.newimg)

        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg - tf.tanh(self.timg) / 2), [1, 2, 3])

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab) * self.output, 1)
        other = tf.reduce_max((1 - self.tlab) * self.output - (self.tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const * loss1)
        self.loss = self.loss1 + self.loss2
        self.grads = tf.reduce_max(tf.gradients(self.loss, [modifier]))

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, X, Y):
        """
        Perform the L_2 attack on the given images for the given targets.

        :param X: samples to generate advs
        :param Y: the original class labels
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        nb_classes = Y.shape[1]

        # random select target class for targeted attack
        y_target = np.copy(Y)
        if self.TARGETED:
            for i in range(Y.shape[0]):
                current = int(np.argmax(Y[i]))
                target = np.random.choice(other_classes(nb_classes, current))
                y_target[i] = np.eye(nb_classes)[target]

        X_adv = np.zeros_like(X)
        for i in tqdm(range(0, X.shape[0], self.batch_size)):
            start = i
            end = i + self.batch_size
            end = np.minimum(end, X.shape[0])
            X_adv[start:end] = self.attack_batch(X[start:end], y_target[start:end])

        return X_adv

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] -= self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # batch_size = self.batch_size
        batch_size = imgs.shape[0]

        # convert to tanh-space
        imgs = np.arctanh(imgs * 1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
        # o_bestattack = np.copy(imgs)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # print(o_bestl2)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg = self.sess.run([self.train, self.loss,
                                                         self.l2dist, self.output,
                                                         self.newimg], feed_dict={K.learning_phase(): 0})

                # print out the losses every 10%
                # if iteration % (self.MAX_ITERATIONS // 10) == 0:
                #     print(iteration, self.sess.run((self.loss, self.loss1, self.loss2, self.grads, self.max_mod), feed_dict={K.learning_phase(): 0}))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        # print('l2:', l2, 'bestl2[e]: ', bestl2[e])
                        # print('score:', np.argmax(sc), 'bestscore[e]:', bestscore[e])
                        # print('np.argmax(batchlab[e]):', np.argmax(batchlab[e]))
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        print('sucess rate: %.4f' % (1-np.sum(o_bestl2==1e10)/self.batch_size))
        return o_bestattack

class CarliniLID:
    def __init__(self, sess, model, image_size, num_channels, num_labels, batch_size=100,
                 confidence=L2_CONFIDENCE, targeted=L2_TARGETED, learning_rate=L2_LEARNING_RATE,
                 binary_search_steps=L2_BINARY_SEARCH_STEPS, max_iterations=L2_MAX_ITERATIONS,
                 abort_early=L2_ABORT_EARLY,
                 initial_const=L2_INITIAL_CONST):
        """
        The modified L_2 optimized attack to break LID detector. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """
        self.model = model
        self.sess = sess
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        shape = (self.batch_size, self.image_size, self.image_size, self.num_channels)

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))
        self.max_mod = tf.reduce_max(modifier)

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((self.batch_size, self.num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (self.batch_size, self.num_labels))
        self.assign_const = tf.placeholder(tf.float32, [self.batch_size])

        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        self.newimg = tf.tanh(modifier + self.timg) / 2

        # prediction BEFORE-SOFTMAX of the model
        self.output = self.model(self.newimg)

        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg - tf.tanh(self.timg) / 2), [1, 2, 3])

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab) * self.output, 1)
        other = tf.reduce_max((1 - self.tlab) * self.output - (self.tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        # add lis loss to the attack
        self.clean_logits = tf.placeholder(tf.float32, (1, self.batch_size, None))
        loss_lid = lid_adv_term(self.clean_logits, self.output, self.batch_size)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const * (loss1 + loss_lid))
        self.loss = self.loss1 + self.loss2
        self.grads = tf.reduce_max(tf.gradients(self.loss, [modifier]))

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, X, Y):
        """
        Perform the L_2 attack on the given images for the given targets.

        :param X: samples to generate advs
        :param Y: the original class labels
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        nb_classes = Y.shape[1]

        # random select target class for targeted attack
        y_target = np.copy(Y)
        if self.TARGETED:
            for i in range(Y.shape[0]):
                current = int(np.argmax(Y[i]))
                target = np.random.choice(other_classes(nb_classes, current))
                y_target[i] = np.eye(nb_classes)[target]

        X_adv = np.zeros_like(X)
        for i in tqdm(range(0, X.shape[0], self.batch_size)):
            start = i
            end = i + self.batch_size
            end = np.minimum(end, X.shape[0])
            X_adv[start:end] = self.attack_batch(X[start:end], y_target[start:end])

        return X_adv

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] -= self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # batch_size = self.batch_size
        batch_size = imgs.shape[0]

        # convert to tanh-space
        imgs = np.arctanh(imgs * 1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
        # o_bestattack = np.copy(imgs)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # print(o_bestl2)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})

            # get clean logits of clean samples:
            c_logits = self.sess.run([self.output], feed_dict={K.learning_phase(): 0})

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg = self.sess.run([self.train, self.loss,
                                                         self.l2dist, self.output,
                                                         self.newimg], feed_dict={K.learning_phase(): 0,
                                                                                  self.clean_logits: c_logits})

                # print out the losses every 10%
                # if iteration % (self.MAX_ITERATIONS // 10) == 0:
                #     print(iteration, self.sess.run((self.loss, self.loss1, self.loss2, self.grads, self.max_mod), feed_dict={K.learning_phase(): 0}))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        # print('l2:', l2, 'bestl2[e]: ', bestl2[e])
                        # print('score:', np.argmax(sc), 'bestscore[e]:', bestscore[e])
                        # print('np.argmax(batchlab[e]):', np.argmax(batchlab[e]))
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        print('sucess rate: %.4f' % (1-np.sum(o_bestl2==1e10)/self.batch_size))
        return o_bestattack


Li_DECREASE_FACTOR = 0.9   # 0<f<1, rate at which we shrink tau; larger is more accurate
Li_MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
Li_ABORT_EARLY = True      # abort gradient descent upon first valid solution
Li_INITIAL_CONST = 1e-5    # the first value of c to start at
Li_LEARNING_RATE = 5e-3    # larger values converge faster to less accurate results
Li_LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
Li_REDUCE_CONST = False    # try to lower c each iteration; faster to set to false
Li_TARGETED = True         # should we target one specific class? or just be wrong?
Li_CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better


class CarliniLi:
    def __init__(self, sess, model,
                 targeted=Li_TARGETED, learning_rate=Li_LEARNING_RATE,
                 max_iterations=Li_MAX_ITERATIONS, abort_early=Li_ABORT_EARLY,
                 initial_const=Li_INITIAL_CONST, largest_const=Li_LARGEST_CONST,
                 reduce_const=Li_REDUCE_CONST, decrease_factor=Li_DECREASE_FACTOR,
                 const_factor=Li_CONST_FACTOR):
        """
        The L_infinity optimized attack.
        Returns adversarial examples for the supplied model.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. Should be set to a very small
          value (but positive).
        largest_const: The largest constant to use until we report failure. Should
          be set to a very large value.
        reduce_const: If true, after each successful attack, make const smaller.
        decrease_factor: Rate at which we should decrease tau, less than one.
          Larger produces better quality results.
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        """
        self.model = model
        self.sess = sess

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor

        self.grad = self.gradient_descent(sess, model)

    def gradient_descent(self, sess, model):
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y

        shape = (1, model.image_size, model.image_size, model.num_channels)

        # the variable to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        tau = tf.placeholder(tf.float32, [])
        simg = tf.placeholder(tf.float32, shape)
        timg = tf.placeholder(tf.float32, shape)
        tlab = tf.placeholder(tf.float32, (1, model.num_labels))
        const = tf.placeholder(tf.float32, [])

        newimg = (tf.tanh(modifier + simg) / 2)

        output = model.predict(newimg)
        orig_output = model.predict(tf.tanh(timg) / 2)

        real = tf.reduce_sum((tlab) * output)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000))

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other)

        # sum up the losses
        loss2 = tf.reduce_sum(tf.maximum(0.0, tf.abs(newimg - tf.tanh(timg) / 2) - tau))
        loss = const * loss1 + loss2

        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        init = tf.variables_initializer(var_list=[modifier] + new_vars)

        def doit(oimgs, labs, starts, tt, CONST):
            # convert to tanh-space
            imgs = np.arctanh(np.array(oimgs) * 1.999999)
            starts = np.arctanh(np.array(starts) * 1.999999)

            # initialize the variables
            sess.run(init)
            while CONST < self.LARGEST_CONST:
                # try solving for each value of the constant
                print('try const', CONST)
                for step in range(self.MAX_ITERATIONS):
                    feed_dict = {timg: imgs,
                                 tlab: labs,
                                 tau: tt,
                                 simg: starts,
                                 const: CONST}
                    if step % (self.MAX_ITERATIONS // 10) == 0:
                        print(step, sess.run((loss, loss1, loss2), feed_dict=feed_dict))

                    # perform the update step
                    _, works = sess.run([train, loss], feed_dict=feed_dict)

                    # it worked
                    if works < .0001 * CONST and self.ABORT_EARLY:
                        get = sess.run(output, feed_dict=feed_dict)
                        works = compare(np.argmax(get), np.argmax(labs))
                        if works:
                            scores, origscores, nimg = sess.run((output, orig_output, newimg), feed_dict=feed_dict)
                            l2s = np.square(nimg - np.tanh(imgs) / 2).sum(axis=(1, 2, 3))

                            return scores, origscores, nimg, CONST

                # we didn't succeed, increase constant and try again
                CONST *= self.const_factor

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        for img, target in zip(imgs, targets):
            r.extend(self.attack_single(img, target))
        return np.array(r)

    def attack_single(self, img, target):
        """
        Run the attack on a single image and label
        """

        # the previous image
        prev = np.copy(img).reshape((1, self.model.image_size, self.model.image_size, self.model.num_channels))
        tau = 1.0
        const = self.INITIAL_CONST

        while tau > 1. / 256:
            # try to solve given this tau value
            res = self.grad([np.copy(img)], [target], np.copy(prev), tau, const)
            if res == None:
                # the attack failed, we return this as our final answer
                return prev

            scores, origscores, nimg, const = res
            if self.REDUCE_CONST: const /= 2

            # the attack succeeded, reduce tau and try again

            actualtau = np.max(np.abs(nimg - img))

            if actualtau < tau:
                tau = actualtau

            print("Tau", tau)

            prev = nimg
            tau *= self.DECREASE_FACTOR
        return prev