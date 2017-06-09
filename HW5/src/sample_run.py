import NN, data_loader, perceptron
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product
from heapq import *


"""
Choose Model
"""
#training_data, test_data = data_loader.load_circle_data()
training_data, test_data = data_loader.load_mnist_data()

#domain = circles
domain = 'mnist'
params, results = [[10, 50, 100], ['tanh', 'relu'], [0.1, 0.01], [10, 50]], []


"""
Cross-validation for Parameter Settings
"""
training_data = np.array(training_data)
length = len(training_data)
index = np.random.permutation(length)
size = length/5
first, second, third, forth, fifth = index[:size], index[size:size*2], index[size*2:size*3], index[size*3:size*4], index[size*4:]
fold_index = [first, second, third, forth, fifth]

for batch_size, activation_function, learning_rate, hidden_layer_width in product(*params):
    accy_all = 0
    net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
    for index in fold_index:
        mask = np.ones(len(training_data), np.bool)
        mask[index] = 0
        hold_out = training_data[index]
        train = training_data[mask]
        print(net.train(train))
        accy_val = net.evaluate(hold_out)
        print(accy_val)
        accy_all += accy_val
    
    accy_all /= 5
    heappush(results, ((1-accy_all), batch_size, activation_function, learning_rate, hidden_layer_width))
    with open("temp.txt", "w") as file:
        for error, batch_size, activation_function, learning_rate, hidden_layer_width in results:
            file.write(str(1-error) + "," + str(batch_size) + "," + str(activation_function) + "," + str(learning_rate) + "," + str(hidden_layer_width) + "\n")


"""
Neural Network
"""
error, batch_size, activation_function, learning_rate, hidden_layer_width = results[0]
net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
learning_curve_n = net.train_with_learning_curve(training_data)
print(net.train(training_data))
print(net.evaluate(test_data))


"""
Perceptron
"""
perc = perceptron.Perceptron(len(training_data[0][0]))
learning_curve_p = perc.train_with_learning_curve(training_data)
print(perc.evaluate(test_data))


"""
Output
"""
with open("result_"+domain+".txt", "w") as file:
    for error, batch_size, activation_function, learning_rate, hidden_layer_width in results:
        file.write(str(1-error) + "," + str(batch_size) + "," + str(activation_function) + "," + str(learning_rate) + "," + str(hidden_layer_width) + "\n")

learning_curve_n = np.array(learning_curve_n)
learning_curve_p = np.array(learning_curve_p)


"""
Plot Learning Curve
"""
# plot learning curve
learning_curve_n = np.array(learning_curve_n)
learning_curve_p = np.array(learning_curve_p)
x = learning_curve_n[:,0]
y_n = learning_curve_n[:,1]
y_p = learning_curve_p[:,1]
r_patch = mpatches.Patch(color='r', label='neural network')
c_patch = mpatches.Patch(color='c', label='perceptron')
plt.xlabel('step')
plt.ylabel('accy')
plt.plot(x, y_n, 'r')
plt.plot(x, y_p, 'c')
plt.legend(handles=[r_patch,c_patch],loc=4)
plt.show()


