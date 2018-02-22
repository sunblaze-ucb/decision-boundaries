## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess
RANDOM_INIT = False

class CarliniL2Ensemble:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST, random_init = RANDOM_INIT,
                 noise_count = 20, noise_mag = 8.):
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

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.random_init = random_init
        self.batch_size = batch_size
        self.noise_count = noise_count
        self.noise_mag = noise_mag

        self.repeat = binary_search_steps >= 10

        shape = (batch_size,image_size,image_size,num_channels)
        
        # initialize noise
        noise_raw = np.random.normal(scale=self.noise_mag, size=(image_size * image_size * num_channels, self.noise_count)).astype(np.float32)
        noisevecs_unit, _ = np.linalg.qr(noise_raw)
        assert noisevecs_unit.shape[1] == self.noise_count
        noisevecs_fv = noisevecs_unit * np.sqrt(image_size * image_size * num_channels) * self.noise_mag
        self.noisevecs = noisevecs_fv.transpose((1, 0)).reshape((self.noise_count, image_size, image_size, num_channels)) # vhwc
        self.noisevecs[self.noise_count - 1] = 0. # %%% keep center?

        # the variable we're going to optimize over
        if self.random_init:
            # fix a set of random initialization for consistent binary search
            # have to rerun the program for different samples
            self.modifier_init = np.random.standard_normal(shape).astype(np.float32)
        else:
            self.modifier_init = np.zeros(shape,dtype=np.float32)
        modifier = tf.Variable(self.modifier_init)

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        # the resulting image, tanh'd to keep bounded from 0 to 255
        self.newimg = tf.tanh(modifier + self.timg) * 127.5 + 127.5
        
        # prediction BEFORE-SOFTMAX of the model
        input_nvhwc = self.newimg[:, None, :, :, :] + self.noisevecs[None, :, :, :, :]
        input_hvhwc = tf.clip_by_value(input_nvhwc, 0., 255.)
        input_nhwc = tf.reshape(input_nvhwc, [batch_size * self.noise_count, image_size, image_size, num_channels])
        output_nl = model.predict(input_nhwc)
        self.output_nvl = tf.reshape(output_nl, [batch_size, self.noise_count, num_labels])
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * 127.5 + 127.5)) / (255 * 255),[1,2,3])
        
        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab[:, None, :])*self.output_nvl,2)
        other = tf.reduce_max((1-self.tlab[:, None, :])*self.output_nvl - (self.tlab[:, None, :]*10000),2)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1_nvl = tf.maximum(0.0, other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1_nvl = tf.maximum(0.0, real-other+self.CONFIDENCE)
        loss1 = tf.reduce_sum(loss1_nvl, axis=1)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2
        
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
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x_vl,y_1h):
            x_v = np.argmax(x_vl - self.CONFIDENCE * y_1h[None, :], axis=1)
            y = np.argmax(y_1h)
            if self.TARGETED:
                return np.all(x_v == y)
            else:
                return np.all(x_v != y)

        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh((imgs - 127.5) / 127.5001)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [False]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # %%% print(o_bestl2)
            print CONST
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
    
            bestl2 = [1e10]*batch_size
            bestscore = [False]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack 
                _, l, l2s, scores_nvl, nimg = self.sess.run([self.train, self.loss, 
                                                         self.l2dist, self.output_nvl, 
                                                         self.newimg])

                # print out the losses every 10%
                if iteration%(self.MAX_ITERATIONS//100) == 0: # %%%
                    print(iteration,self.sess.run((self.loss,self.loss1,self.loss2)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        print 'aborting early at iteration', iteration
                        break
                    prev = l

                # adjust the best result found so far
                for e,(l2,sc_vl,ii) in enumerate(zip(l2s,scores_nvl,nimg)):
                    if l2 < bestl2[e] and compare(sc_vl, batchlab[e]):
                        bestl2[e] = l2
                        bestscore[e] = True
                    if l2 < o_bestl2[e] and compare(sc_vl, batchlab[e]):
                        o_bestl2[e] = l2
                        o_bestscore[e] = True
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if bestscore[e]:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
