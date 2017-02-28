"""
 @author     : Bangqi Wang (bwang.will@gmail.com)
 @file       : main.py
 @brief      : Main file for homework 3
"""

from gen import gen
from perceptron import Perceptron
from perceptron_margin import Perceptron_Margin
from winnon import Winnon
from winnon_margin import Winnon_Margin
from adagrad import AdaGrad
import random
import numpy as np
import matplotlib.pyplot as plt


def draw_1(size, l, m, n, error_p, error_pm, error_w, error_wm, error_a):
    """ drawing plots for question 1.
    """
    f, ax = plt.subplots()
    vertices = np.arange(0, size, 500)
    ax.plot(vertices, error_p, 'b', label='Perceptron')
    ax.plot(vertices, error_pm, 'g', label='Perceptron_Margin')
    ax.plot(vertices, error_w, 'r', label='Winnon')
    ax.plot(vertices, error_wm, 'k', label='Winnon_Margin')
    ax.plot(vertices, error_a, 'c', label='AdaGrad')
    plt.xlabel('N')
    plt.ylabel('W')
    plt.title('Cumulative Number of Mistakes: l = {}, m = {}, n = {}'.format(l, m, n))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


def draw_2(l, m, R, error_p, error_pm, error_w, error_wm, error_a):
    """ drawing plots for question 2.
    """
    f, ax = plt.subplots()
    vertices = np.arange(40, 240, 40)
    ax.plot(vertices, error_p, 'b', label='Perceptron')
    ax.plot(vertices, error_pm, 'g', label='Perceptron_Margin')
    ax.plot(vertices, error_w, 'r', label='Winnon')
    ax.plot(vertices, error_wm, 'k', label='Winnon_Margin')
    ax.plot(vertices, error_a, 'c', label='AdaGrad')
    plt.xlabel('n')
    plt.ylabel('W')
    plt.title('Cumulative Number of Mistakes: l = {}, m = {}, R = {}'.format(l, m, R))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


def draw_bonus_error(error):
    """ drawing plots for bonus problem.
    """
    f, ax = plt.subplots()
    vertices = np.arange(10, 50)
    ax.plot(vertices, error[10:], 'b', label='Error')
    plt.xlabel('Rounds')
    plt.ylabel('Misclassification Error')
    plt.title('Misclassification Error: l = 10, m = 20, n = 40')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


def draw_bonus_loss(loss):
    """ drawing plots for bonus problem.
    """
    f, ax = plt.subplots()
    vertices = np.arange(10, 50)
    ax.plot(vertices, loss[10:], 'b', label='Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Hinge Loss')
    plt.title('Hinge Loss: l = 10, m = 20, n = 40')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


def part_1(l, m, n, size, noise):
    """ function for question 1.
        1. initialize variables
        2. divide training & test dataset
        3. build online learning algorithms
        4. draw plots
    """
    # initilize variables
    (y, x) = gen(l, m, n, size, noise)
    samples, loops, R = size / 10, 20, None
    lrs = [1.5, 0.25, 0.03, 0.005, 0.001]
    alphas = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    gammas = [2.0, 0.3, 0.04, 0.006, 0.001]
    
    # divide dataset: 10% - training, 10% - test.
    left, right = random.sample(range(10),2)
    x_train, x_test = x[left*samples:(left+1)*samples], x[right*samples:(right+1)*samples]
    y_train, y_test = y[left*samples:(left+1)*samples], y[right*samples:(right+1)*samples]
    
    # build online learning algorithms
    perceptron = Perceptron(R, x_train, x_test, y_train, y_test, n, samples, size)
    perceptron_margin = Perceptron_Margin(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, lrs)
    winnon = Winnon(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, alphas)
    winnon_margin = Winnon_Margin(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, alphas, gammas)
    adagrad = AdaGrad(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, lrs)
    
    # draw plots
    draw_1(size, l, m, n, perceptron.error, perceptron_margin.error, winnon.error, winnon_margin.error, adagrad.error)


def part_2(R):
    """ function for question 1.
        1. initialize variables
        2. loop for n in [40, 80, 120, 160, 200]
        3. for each n value, divide training & test dataset
        4. build online learning algorithms
        5. count the number of mistakes made when get R correct predictions.
        6. draw plots
    """
    # initialize variables
    l, m, size, noise = 10, 20, 50000, False
    samples, loops = size / 10, 20
    lrs = [1.5, 0.25, 0.03, 0.005, 0.001]
    alphas = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    gammas = [2.0, 0.3, 0.04, 0.006, 0.001]
    error_p, error_pm, error_w, error_wm, error_a = [], [], [], [], []

    # loop for n
    for n in range(40, 240, 40):
        print 'n = {}'.format(n)
        # divide dataset: 10% - training, 10% - test. (fixed random seed)
        (y, x) = gen(l, m, n, size, noise)
        random.seed(1)
        left, right = random.sample(range(10),2)
        x_train, x_test = x[left*samples:(left+1)*samples], x[right*samples:(right+1)*samples]
        y_train, y_test = y[left*samples:(left+1)*samples], y[right*samples:(right+1)*samples]
        
        # build online learning algorithms, and count mistakes made
        perceptron = Perceptron(R, x_train, x_test, y_train, y_test, n, samples, size)
        error_p.append(perceptron.mistakes)
        # [517, 607, 605, 595, 611]
        perceptron_margin = Perceptron_Margin(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, lrs)
        error_pm.append(perceptron_margin.mistakes)
        # [737, 684, 640, 699, 603]
        winnon = Winnon(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, alphas)
        error_w.append(winnon.mistakes)
        # [118, 235, 300, 352, 378]
        winnon_margin = Winnon_Margin(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, alphas, gammas)
        error_wm.append(winnon_margin.mistakes)
        # [559, 615, 305, 349, 374]
        adagrad = AdaGrad(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, lrs)
        error_a.append(adagrad.mistakes)
        # [583, 543, 527, 548, 601]

    # draw plots
    draw_2(l, m, R, error_p, error_pm, error_w, error_wm, error_a)


def part_3(l, m, n):
    """ function for question 3.
        1. initialize variables
        2. generate training & test dataset
        3. build online learning algorithms
    """
    # initilize variables
    size = 50000
    samples, loops, R = size / 10, 20, None
    lrs = [1.5, 0.25, 0.03, 0.005, 0.001]
    alphas = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    gammas = [2.0, 0.3, 0.04, 0.006, 0.001]
    
    # divide dataset: 10% - training, 10% - test.
    # y, x = [], []
    (y, x) = gen(l, m, n, 10, False)
    (y_train, x_train) = gen(l, m, n, 50000, True)
    (y_test, x_test) = gen(l, m, n, 10000, False)
    
    # build online learning algorithms
    perceptron = Perceptron(R, x_train, x_test, y_train, y_test, n, size, size)
    perceptron_margin = Perceptron_Margin(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, lrs)
    winnon = Winnon(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, alphas)
    winnon_margin = Winnon_Margin(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, alphas, gammas)
    adagrad = AdaGrad(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, lrs)
    

def bonus(l, m, n):
    """ function for question 3.
        1. initialize variables
        2. generate training & test dataset
        3. build online learning algorithms
    """
    # initialize variables
    lrs = [1.5, 0.25, 0.03, 0.005, 0.001]
    (y, x) = gen(l, m, n, 10000, True) 
    # placeholder 
    (y_train, x_train) = gen(l, m, n, 1, True)
    (y_test, x_test) = gen(l, m, n, 1, False)
    size, samples, loops, R = 10000, 10000, 50, None
    
    # build online learning algorithms
    adagrad = AdaGrad(R, x, x_train, x_test, y, y_train, y_test, n, samples, size, loops, lrs)
    
    # draw plots
    draw_bonus_error(adagrad.error)
    draw_bonus_loss(adagrad.loss)
    


if __name__ == '__main__':
    """ Main Function:
        uncomment the function to run the corresponding part.
    """
    # part_1(10, 100, 500, 50000, False)
    # part_1(10, 100, 1000, 50000, False)
    # part_2(1000)
    # part_3(10, 100, 1000)
    # part_3(10, 500, 1000)
    # part_3(10, 1000, 1000)
    # bonus(10, 20, 40)

