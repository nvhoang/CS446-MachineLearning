"""
 @author     : Bangqi Wang (bwang.will@gmail.com)
 @file       : perceptron.py
 @brief      : Implementation for perceptron
"""
import numpy as np

class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self, R, x_train, x_test, y_train, y_test, n, samples, size, loops=1, lr=1, gamma=0):
        super(Perceptron, self).__init__()
        """ init function:
            1. parse dataset
            2. train
            3. test
            4. count
        """
        # parse dataset
        shuffle_samples = np.random.permutation(samples)
        self.x_train = x_train[shuffle_samples][:]
        self.x_test = x_test
        self.y_train = np.array([y_train[shuffle_samples]]).T
        self.y_test = np.array([y_test]).T
        
        # train
        if not R:
            self.training(lr, gamma, loops, n, size)
        # count
        else:
            self.count(R, lr, gamma, loops, n, size)


    def training(self, lr, gamma, loops, n, size):
        """ train
            this function updates w and theta with training dataset
        """
        self.w, self.theta, self.mistakes, self.error = np.zeros(n), 0, 0, []
        for loop in range(loops):
            for i in range(size):
                x, y = self.x_train[i], self.y_train[i]
                # updates rule
                if (y * (np.dot(self.w, x) + self.theta))[0] <= gamma:
                    self.w += lr * y * x
                    self.theta += lr * y
                    self.mistakes += 1
                if i % 500 == 0:
                    self.error.append(self.mistakes)
        acc = self.evaluation()
        print 'acc = {}'.format(acc)


    def evaluation(self):
        """ evaluate:
            this function calculate the accuracy of test dataset
        """
        prediction = self.theta + np.dot(self.x_test, self.w)
        for i in range(10000):
            prediction[i] = 1 if prediction[i] >= 0 else -1
        return np.sum(np.equal(prediction.astype(np.int32), self.y_test.flat))/float(10000)


    def count(self, R, lr, gamma, loops, n, size):
        """ count:
            this function count the number of mistakes made when get R straight correct prediction 
        """
        self.w, self.theta, self.mistakes, self.correct = np.zeros(n), 0, 0, 0
        for loop in range(loops):
            for i in range(size):
                x, y = self.x[i], self.y[i]
                if (y * (np.dot(self.w, x) + self.theta))[0] <= gamma:
                    self.w += lr * y * x
                    self.theta += lr * y
                    self.mistakes += 1
                    self.correct = 0
                else:
                    self.correct += 1
                    if self.correct == R:
                        return


