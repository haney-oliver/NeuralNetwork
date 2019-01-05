import numpy as np
import matplotlib.pyplot as plt

## author: haney-oliver
## simple neural network to explore basic back-propogation
## note: this is my first ever python project
class NeuralNetwork():
    def __init__(self):

        #### FIELDS ####
        #-weights
        self.theta1 = 2 * (np.random.random((3 , 1))) - 1

        #-predictions
        self.predictions = np.array([[]])


    #### BEHAVIOR ####
    #-sigmoid activation fucntion
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #-sigmoid derivative
    def sigmoid_derivative(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    #-training function
    def train(self, inputs, outputs, number_loops):
        for i in range(number_loops):
            learning_rate = .15
            input_layer = inputs
            self.predictions = self.think(inputs)
            error = (training_data_o - self.predictions)
            adjustments = error * self.sigmoid_derivative(self.predictions)
            self.theta1 +=(np.dot(input_layer.T, adjustments) * learning_rate)

    #-make predictions
    def think(self, inputs):
        inputs = inputs.astype(float)
        guess = self.sigmoid(np.dot(inputs, self.theta1))
        return guess

    #-data visualization
    def visualize(self):
        x = np.linspace(-4, 4, 100)
        plt.ylim(0, 1)
        plt.plot(x, self.sigmoid(x), label='sigmoid')
        plt.plot(x, self.sigmoid_derivative(x), label='sigmoid_derivative')
        plt.title("Sigmoid and its Derivative")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    #### CONSTRUCTOR ####
    neural_network = NeuralNetwork()

    #-data sets
    ## relationship: if i[0-8][0] == 1, o = 1
    training_data_o = np.array([[0,0,1,1]]).T

    training_data_i = np.array([[0,1,0],
                                [0,0,1],
                                [1,0,0],
                                [1,1,1]])


    #### MAIN ####
    print('Weights before training :  ')
    print(neural_network.theta1)
    print('Predictions before training :  ')
    print(neural_network.think(training_data_i))

    neural_network.train(training_data_i, training_data_o, 10000)

    print()
    print()
    print('Weights after training :  ')
    print(neural_network.theta1)
    print('Predictions after training :  ')
    print(neural_network.predictions)

    neural_network.visualize()

    while input != "x":
        print()
        print('Enter a new situation :  ')

        x0 = str(input("Input 1 :  "))
        x1 = str(input("Input 2 :  "))
        x2 = str(input("Input 3 :  "))

        print(f'Input data :  {x0} {x1} {x2}')
        print()
        print('Prediction of new situation :  ')
        print(neural_network.think(np.array([x0, x1, x2])))
