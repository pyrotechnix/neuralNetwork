import numpy
import numpy as np
import os
from matplotlib import pyplot as plt
import csv


# these functions are kinda useless as I rewrote them after finding out that the built-in methods were way better
def matrixBroadcast(m, broadcastLen):
    returnMatrix = np.zeros((len(m), broadcastLen))
    for i in range(0, len(m)):
        returnMatrix[i] = (np.full(broadcastLen, m[i][0]))
    return returnMatrix


# Activation function that neural network uses (Currently leaky relu)
def activationFunction(n):
    return np.maximum(-n * 0.01, n)


# Derivative of activation function when passed the activated value
def activationDerivative(n):
    output_derivative = np.full_like(n, 0.01)
    output_derivative = np.where(n > 0, 1.0, output_derivative)
    return output_derivative


class neuralNetwork:
    # takes inputNum as integer, outputNum as integer, layerInformation as list of ints detailing how many neurons
    # per layer
    inputCount = None
    # defining variables to help keep track of the number of inputs and outputs, these are just defined with
    # layerInformation anyway, so it's not really that important
    _weightMatrix = None
    # shouldn't need a matrix of neurons since the vector multiplication should output the value for the neurons anyway
    _biasMatrix = None
    _trainingData = None
    _trainingLabels = None
    _testingData = None
    _testingLabels = None
    _currentOutput = None
    _activationValues = None
    _catagoryDict = None

    def __init__(self, layerInformation):
        if len(layerInformation) < 3:
            print("Invalid, no hidden layer")
        else:
            self.inputCount = layerInformation[0]
            self._weightMatrix = []
            self._biasMatrix = []
            self._activationValues = []

            # initialising random weights
            for i in range(1, len(layerInformation)):
                weightLayer = np.random.randn(layerInformation[i], layerInformation[i - 1]) / 10
                self._weightMatrix.append(weightLayer)

            # initialising random biases
            for i in range(1, len(layerInformation)):
                biasLayer = np.random.randn(layerInformation[i]) / 10
                self._biasMatrix.append(biasLayer.reshape(-1, 1))

            for i in range(0, len(layerInformation)):
                layer = np.zeros(layerInformation[i])
                self._activationValues.append(layer.reshape(-1, 1))

    # trainingData should be only a 2 dimensional array, as it is converted into a 2 dimensional matrix later
    def setTrainingData(self, trainingData, trainingLabels, testingData, testingLabels):
        self._trainingData = np.zeros([len(trainingData), len(trainingData[0].flatten())])
        self._testingData = np.zeros([len(testingData), len(trainingData[0].flatten())])
        for i in range(0, len(self._trainingData)):
            self._trainingData[i] = trainingData[i].flatten()
        for i in range(0, len(self._testingData)):
            self._testingData[i] = testingData[i].flatten()
        # diving all values by the max to keep them in between 0 and 1 instead of making them big, since the
        # numpy exp function doesn't like large numbers, and its considered good practice anyway
        # Normalizing i guess you could call it
        self._trainingData /= self._trainingData.max()
        self._testingData /= self._testingData.max()
        self.interpretCatagories(trainingLabels)
        self._trainingLabels = np.zeros([len(trainingData), len(self._catagoryDict), 1])
        for i in range(0, len(trainingLabels)):
            self._trainingLabels[i] = self._catagoryDict[str(trainingLabels[i])]
        self._testingLabels = np.zeros([len(testingData), len(self._catagoryDict), 1])
        for i in range(0, len(testingLabels)):
            self._testingLabels[i] = self._catagoryDict[str(testingLabels[i])]

    # converts labels to a list of catagories, and assigns an output neuron for each
    # this assumes that the number of output neurons is already correct
    def interpretCatagories(self, labels):
        catagories = []
        for i in labels:
            if i not in catagories:
                catagories.append(i)
        catagoryCount = len(catagories)
        for i in range(0, len(catagories)):
            catagories[i] = str(catagories[i])
        self._catagoryDict = {}
        for i in range(0, catagoryCount):
            addArr = np.array([np.zeros(catagoryCount)])
            addArr[0][i] = 1
            self._catagoryDict[catagories[i]] = addArr.T
        return self._catagoryDict

    # Debug function to set values of weights and biases if you want to hand pick the values for some reason
    def setValues(self, weights, biases):
        print(self._weightMatrix)
        self._weightMatrix = weights
        print(self._weightMatrix)
        print(self._biasMatrix)
        self._biasMatrix = biases
        print(self._biasMatrix)

    # inputs taken as list of arrays, since it converts the arrays to 2d matrices later anyway
    def feedForward(self, inputs):
        if len(inputs[0]) != self.inputCount:
            print('invalid operation, number of inputs does not match')
        else:
            currentLayer = np.array(inputs).T
            # print(currentLayer)
            self._activationValues[0] = currentLayer
            for i in range(0, len(self._weightMatrix) - 1):
                # multiplying and adding weights
                currentLayer = self._weightMatrix[i] @ currentLayer
                # adding biases
                currentLayer = matrixBroadcast(self._biasMatrix[i], len(currentLayer[0])) + currentLayer
                # Apply activation function to results
                currentLayer = activationFunction(currentLayer)
                self._activationValues[i + 1] = currentLayer
            currentLayer = self._weightMatrix[len(self._weightMatrix) - 1] @ currentLayer
            # adding biases
            currentLayer = matrixBroadcast(self._biasMatrix[len(self._biasMatrix) - 1],
                                           len(currentLayer[0])) + currentLayer
            # softmax function
            maxPerSample = np.max(currentLayer, axis=0, keepdims=True)
            # employing a technique to prevent overflow where you shift untreated outputs down by the max value
            # this is supposed to prevent overflow and underflow
            shiftedCurrentLayer = currentLayer - maxPerSample
            # creating the denominators for the softmax function by column
            summedDenominators = np.sum(np.exp(shiftedCurrentLayer), axis=0, keepdims=True)
            currentLayer = np.exp(shiftedCurrentLayer) / summedDenominators
            self._currentOutput = currentLayer
            self._activationValues[-1] = currentLayer
            # Returning confidence scores for each layer
            return np.max(currentLayer, axis=0)

    # calculate the loss using multi class cross entropy
    def calculateLoss(self, expected):
        losses = np.hstack(expected) * np.log(self._currentOutput)
        print("Total loss:")
        print(np.sum(losses))
        return -np.sum(losses) / len(losses[0])

    def backpropagate(self, expected):
        learningRate = 0.01
        weightChanges = []
        biasChanges = []
        outputDerivatives = (self._currentOutput - np.hstack(expected)) / len(self._currentOutput[0])
        # giving us (actual - expected) / n
        outputWeightDerivatives = outputDerivatives @ self._activationValues[-2].T
        weightChanges.append(outputWeightDerivatives)
        outputBiasDerivatives = np.sum(outputDerivatives, axis=1, keepdims=True)
        biasChanges.append(outputBiasDerivatives)
        for i in range(1, len(self._weightMatrix)):
            # finding error signal for current layer
            d_prevZ = self._weightMatrix[-i].T @ outputDerivatives
            d_prevZ = d_prevZ * activationDerivative(self._activationValues[-i - 1])
            # using error signals to calculate weight and bias derivatives
            weightDerivatives = d_prevZ @ self._activationValues[-i - 2].T
            weightChanges.append(weightDerivatives)
            biasChanges.append(np.sum(d_prevZ, axis=1, keepdims=True))
            outputDerivatives = d_prevZ
            # making changes to weight and biases
        for i in range(0, len(self._weightMatrix)):
            self._weightMatrix[i] -= weightChanges[-1 - i] * learningRate
        for i in range(0, len(self._biasMatrix)):
            self._biasMatrix[i] -= biasChanges[-1 - i] * learningRate

    # Train function, simply combines all backpropagation and feed forward into a high-level function that
    # handles stuff like batches, epochs and stuff like that
    def train(self, epochNum):
        batchSize = 100
        loopCount = int(len(self._trainingData) / batchSize)
        for i in range(0, epochNum):
            for j in range(0, loopCount):
                self.feedForward(self._trainingData[j * batchSize:(j + 1) * batchSize])
                self.backpropagate(self._trainingLabels[j * batchSize:(j + 1) * batchSize])
            if len(self._trainingData) % batchSize != 0:
                self.feedForward((self._trainingData[-(len(self._trainingData) % batchSize):-1]))
                self.backpropagate((self._trainingLabels[-(len(self._trainingLabels) % batchSize):-1]))
            print(f"epoch number {i} completed")
            if i % 100 == 0:
                print(f"for epoch number {i}:")
                self.testAccuracy()

    # Testing the accuracy of the network
    def testAccuracy(self):
        correctCount = 0
        self.feedForward(self._testingData)
        checkArray = np.argmax(self._activationValues[-1], axis=0)
        checkArray -= np.argmax(np.hstack(self._testingLabels), axis=0)
        for i in checkArray:
            if i == 0:
                correctCount += 1
        print(f"Unseen accuracy = {correctCount / len(self._testingData) * 100}%")
        correctCount = 0
        self.feedForward(self._trainingData)
        checkArray = np.argmax(self._activationValues[-1], axis=0)
        checkArray -= np.argmax(np.hstack(self._trainingLabels), axis=0)
        for i in checkArray:
            if i == 0:
                correctCount += 1
        print(f"Training data accuracy = {correctCount / len(self._trainingData) * 100}%")

    # displaying the results of the feed forward predictions using matplotlib
    def displayAnswers(self):
        reverseDict = {}
        confidences = self.feedForward(self._testingData)
        print(confidences)
        answerKey = np.argmax(self._activationValues[-1], axis=0)
        for i in self._catagoryDict:
            reverseDict[np.argmax(self._catagoryDict[i])] = i
        idx = 0
        showList = []
        while len(showList) < 50 or idx < 500:
            # if answerKey[idx] != np.argmax(self._testingLabels[idx]):
            showList.append(idx)
            idx += 1

        f, subpl = plt.subplots(5, 10)
        for i in range(0, 5):
            for j in range(0, 10):
                subpl[i][j].imshow(self._testingData[i * 10 + j].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
                currentConfidence = f"{(confidences[i * 10 + j]):.{3}f}"
                subpl[i][j].set_title(reverseDict[answerKey[i * 10 + j]] + "\n" + currentConfidence)
                subpl[i][j].axis('off')
        plt.show()

    # Saves network to a file so that it can be reloaded later
    def saveNetwork(self, path):
        print("Saving network...")
        print("saving weights...")
        with open(path + "_weights", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in self._weightMatrix:
                writer.writerow(i.shape)
                writer.writerow(i.flatten())

        print("saving biases...")
        with open(path + "_biases", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in self._biasMatrix:
                writer.writerow(i.flatten())
        print("Saved!")

    # Loading saved network
    # NOTE: Assumes that the network doesn't have more layers than the loaded network, since if it does
    # it will not fully overwrite the random network. Should not be an issue
    def loadNetwork(self, path):
        # setting the weights
        print("Loading weights...")
        with open(path + "_weights", 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            weightArr = []
            for row in reader:
                print(np.array(row))
                weightArr.append(np.array(row).astype(float))
            for i in range(0, len(self._weightMatrix)):
                self._weightMatrix[i] = np.reshape(weightArr[2 * i + 1], tuple(weightArr[2 * i].astype(int)))
            print("Loaded weights")

        print("Loading biases...")
        with open(path + "_biases", 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            biasArr = []
            for row in reader:
                biasArr.append(np.array(row).astype(float))
            for i in range(0, len(self._biasMatrix)):
                self._biasMatrix[i] = biasArr[i].reshape(-1, 1)
        print("Loaded biases")
        print("Loaded!")
