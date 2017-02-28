"""
 @author     : Bangqi Wang (bwang.will@gmail.com)
 @file       : perceptron_margin.py
 @brief      : Implementation for perceptron with margin
"""
import numpy as np
import Queue

class Perceptron_Margin(object):
    """docstring for Perceptron_Margin"""
    def __init__(self, R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, lrs, gamma=1):
        super(Perceptron_Margin, self).__init__()
        """ init function:
            1. parse dataset
            2. train
            3. test
            4. count
        """
        # parse x, y entire dataset
        shuffle_size = np.random.permutation(size)
        self.x = x[shuffle_size][:]
        self.y = np.array([y[shuffle_size]]).T
        # parse train & test dataset
        shuffle_samples = np.random.permutation(samples)
        self.x_train = x_train[shuffle_samples][:]
        self.x_test = x_test
        self.y_train = np.array([y_train[shuffle_samples]]).T
        self.y_test = np.array([y_test]).T

        # train
        self.training(lrs, n, samples, loops, gamma)
        if not R:
            # validation
            self.validation(gamma, n, size)
        else:
            # count
            self.count(R, gamma, n, size)


    def training(self, lrs, n, samples, loops, gamma):
        """ train
            this function tunes the lr, and choose the best lr from lrs
        """
        queue = Queue.PriorityQueue()
        # calculate accuracy for each lr
        for lr in lrs:
            w, theta, mistakes, error = np.zeros(n), 0, 0, []
            for loop in range(loops):
                for i in range(samples):
                    x, y = self.x_train[i], self.y_train[i]
                    # made misteke
                    if (y * (np.dot(w, x) + theta))[0] < 0:
                        mistakes += 1
                    # update rules
                    if (y * (np.dot(w, x) + theta))[0] <= gamma:
                        w += lr * y * x
                        theta += lr * y
                    if i % 500 == 0:
                        error.append(mistakes)
            # test
            acc = self.evaluation(w, theta, samples)
            print 'lr = {}, acc = {}'.format(lr, acc)
            queue.put([-acc, lr, w, theta, error])
        # choose the best parameters
        self.loss, self.lr = queue.get()[:2]
        print '===Best===> lr: {} -> acc: {}'.format(self.lr, -self.loss)


    def evaluation(self, w, theta, samples):
        """ evaluate:
            this function calculate the accuracy of test dataset
        """
        prediction = theta + np.dot(self.x_test, w)
        for i in range(samples):
            prediction[i] = 1 if prediction[i] >= 0 else -1
        return np.sum(np.equal(prediction.astype(np.int32), self.y_test.flat))/float(samples)


    def validation(self, gamma, n, size, loops=1):
        """ validation:
            this function test on entire dataset after tuning lr
        """
        print '===Best===> lr: {}'.format(self.lr)
        self.w, self.theta, self.mistakes, self.error = np.zeros(n), 0, 0, []
        for loop in range(loops):
            for i in range(size):
                x, y = self.x[i], self.y[i]
                # check mistake
                if (y * (np.dot(self.w, x) + self.theta))[0] < 0:
                    self.mistakes += 1
                # update rules
                if (y * (np.dot(self.w, x) + self.theta))[0] <= gamma:
                    self.w += self.lr * y * x
                    self.theta += self.lr * y
                if i % 500 == 0:
                    self.error.append(self.mistakes)


    def count(self, R, gamma, n, size, loops=1):
        """ count:
            this function count the number of mistakes made when get R straight correct prediction 
        """
        print '===count===> Best lr: {}'.format(self.lr)
        self.w, self.theta, self.mistakes, self.correct = np.zeros(n), 0, 0, 0
        for loop in range(loops):
            for i in range(size):
                x, y = self.x[i], self.y[i]
                # reset counter if made mistake
                if (y * (np.dot(self.w, x) + self.theta))[0] < 0:
                    self.mistakes += 1
                    self.correct = 0
                # if predict correct R times, return
                else:
                    self.correct += 1
                    if self.correct == R:
                        return
                # update rules
                if (y * (np.dot(self.w, x) + self.theta))[0] <= gamma:
                    self.w += self.lr * y * x
                    self.theta += self.lr * y
        

