import numpy
import numpy as np
import os
from matplotlib import pyplot as plt
import csv


# these functions are kinda useless as I rewrote them after finding out that the built-in methods were way better
def matrixMultiply(a, b):
    if len(a[0]) != len(b):
        print("Invalid!")
        return None
    else:
        return a @ b


def matrixAdd(a, b):
    if len(a) != len(b):
        print("Invalid!")
        return None
    else:
        return a + b


def matrixBroadcast(m, broadcastLen):
    returnMatrix = np.zeros((len(m), broadcastLen))
    for i in range(0, len(m)):
        returnMatrix[i] = (np.full(broadcastLen, m[i][0]))
    return returnMatrix


def sigmoid(n):
    return 1 / (1 + np.exp(-1 * n))


def sigmoidDerivative(n):
    return sigmoid(n) * (1 - sigmoid(n))


def sigmoidDerivativeBetter(sigNum):
    return sigNum * (1 - sigNum)


def activationFunction(n):
    return np.maximum(-n * 0.01, n)


def activationDerivative(n):

    output_derivative = np.full_like(n, 0.01)

    # For elements where arr > 0, set the derivative to 1.
    # np.where is efficient for element-wise conditional operations.
    output_derivative = np.where(n > 0, 1.0, output_derivative)
    return output_derivative



class neuralNetwork:
    # takes inputNum as integer, outputNum as integer, layerInformation as list of ints detailing how many neurons
    # per layer
    inputCount = None
    outputCount = None
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
            self.outputCount = layerInformation[-1]
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

    # trainingData should be only a 2 dimensional array, as it is converted into a 2 dimensional vector later
    def setTrainingData(self, trainingData, trainingLabels, testingData, testingLabels):
        self._trainingData = np.zeros([len(trainingData), len(trainingData[0].flatten())])
        self._testingData = np.zeros([len(testingData), len(trainingData[0].flatten())])
        for i in range(0, len(self._trainingData)):
            self._trainingData[i] = trainingData[i].flatten()
        for i in range(0, len(self._testingData)):
            self._testingData[i] = testingData[i].flatten()
        # diving all values by the max to keep them in between 0 and 1 instead of making them big, since the
        # numpy exp function doesn't like large numbers, and its considered good practice anyway
        self._trainingData /= self._trainingData.max()
        self._testingData /= self._testingData.max()
        self.interpretCatagories(trainingLabels)
        self._trainingLabels = np.zeros([len(trainingData), len(self._catagoryDict), 1])
        for i in range(0, len(trainingLabels)):
            self._trainingLabels[i] = self._catagoryDict[str(trainingLabels[i])]
        self._testingLabels = np.zeros([len(testingData), len(self._catagoryDict), 1])
        for i in range(0, len(testingLabels)):
            self._testingLabels[i] = self._catagoryDict[str(testingLabels[i])]

    #converts labels to a list of catagories, and assigns an output neuron for each
    #this assumes that the number of output neurons is already correct
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
                currentLayer = matrixMultiply(self._weightMatrix[i], currentLayer)
                # adding biases
                currentLayer = matrixAdd(matrixBroadcast(self._biasMatrix[i], len(currentLayer[0])), currentLayer)
                # sigmoid results
                currentLayer = activationFunction(currentLayer)
                self._activationValues[i + 1] = currentLayer
                # print(f"layer {i + 1}: {currentLayer}\n-------")
            #currentLayer = matrixMultiply(self._weightMatrix[len(self._weightMatrix) - 1], currentLayer)
            currentLayer = self._weightMatrix[len(self._weightMatrix) - 1] @ currentLayer
            # adding biases
            currentLayer = matrixAdd(matrixBroadcast(self._biasMatrix[len(self._biasMatrix) - 1], len(currentLayer[0])), currentLayer)
            #softmax function
            maxPerSample = np.max(currentLayer, axis=0, keepdims=True)
            #employing a technique to prevent overflow where you shift untreated outputs down by the max value
            #this is supposed to prevent overflow and underflow
            shiftedCurrentLayer = currentLayer - maxPerSample
            #creating the denominators for the softmax function by column
            summedDenominators = np.sum(np.exp(shiftedCurrentLayer), axis=0, keepdims=True)
            currentLayer = np.exp(shiftedCurrentLayer) / summedDenominators
            self._currentOutput = currentLayer
            self._activationValues[-1] = currentLayer
            #setting last layer of activation values too just in case
            # print(f"activation values:\n{self._activationValues}")
            return np.max(currentLayer, axis=0)

    #calculate the loss using MULTI ClASS CROSS ENTROPY 😨😨😨
    def calculateLoss(self, expected):
        losses = np.hstack(expected) * np.log(self._currentOutput)
        print("Total loss:")
        print(np.sum(losses))
        return -np.sum(losses) / len(losses[0])

    def backpropagate(self, expected):
        learningRate = 0.01
        weightChanges = []
        biasChanges = []
        #giving us (actual - expected) / n
        outputDerivatives = (self._currentOutput - np.hstack(expected)) / len(self._currentOutput[0])
        #the / n part is important for later as it uses matrix multiplication to calcualte averages
        #otherwise it would have to multiply by 1/n later, but since we do it here it doesn't have to
        #activation values for the second last layer
        #this part is really cool and handles the averages automatically:
        #since the matrix multiplication is the dot prod of rows and columns
        #output derivatives is a C * B matrix, where B is the batch size
        #self._activationValues[-2] is the second last layer of activation values, and is size A * B
        #self._weights[-1] is the last layer of weights, and is size C * A
        #we want a matrix that is size C * A, so we multiply the C * B matrix by a B * A matrix
        #This can be achieved by multiplying derivatives * the transposition of the activation values
        #This also works because when doing this, the resulting numbers are the dot product between
        #the weight derivatives for that class, and all activations of that value in the previous layer
        #dot producting all of these will give the average derivative per weight over each weight, since
        #derivative of weight = derivative of output class * activation value, and for each weight, you sum
        #the derivatives for the weight with all the activations and sum them together
        #then it is already divided by 1/n, since that was done before
        #this does work I understand it and if I forget it thats ok because it works
        outputWeightDerivatives = outputDerivatives @ self._activationValues[-2].T
        weightChanges.append(outputWeightDerivatives)
        #this is more or less the same, but since the biases are just adjusted by an amount, you can simply
        #sum the derivatives to get an overall value that the biases should be adjusted by
        outputBiasDerivatives = np.sum(outputDerivatives, axis=1, keepdims=True)
        biasChanges.append(outputBiasDerivatives)
        #fuck bro ts genius asf
        #these are the output derivatives for the last layer.
        #next goal is to find error signals of previous activation values.
        #you can do this in the same way as the weights, just in reverse, since the derivative of the cost with respect
        #to the activation values will just be the weight, which will be averaged in the same way.
        for i in range(1, len(self._weightMatrix)):
            d_prevZ = self._weightMatrix[-i].T @ outputDerivatives
            d_prevZ = d_prevZ * activationDerivative(self._activationValues[-i - 1])
            """print(f"activations for layer {len(self._weightMatrix) - i}")
            print(self._activationValues[-i - 1])
            print(f"errors for layer {len(self._weightMatrix) - i}")
            print(d_prevZ)
            print(f"activation values for layer: {len(self._weightMatrix) - i - 1}")
            print(self._activationValues[-i - 2])
            print(f"changes of weights for layer {len(self._weightMatrix) - i}")
            """
            weightDerivatives = d_prevZ @ self._activationValues[-i - 2].T
            weightChanges.append(weightDerivatives)
            biasChanges.append(np.sum(d_prevZ, axis=1, keepdims=True))
            outputDerivatives = d_prevZ
        for i in range(0, len(self._weightMatrix)):
            self._weightMatrix[i] -= weightChanges[-1 - i] * learningRate
        for i in range(0, len(self._biasMatrix)):
            self._biasMatrix[i] -= biasChanges[-1 - i] * learningRate

    def formalCalculatingLossAndStuff(self, sampleCount):
        self.feedForward(self._trainingData[:sampleCount])
        self.backpropagate(self._trainingLabels[:sampleCount])


    def train(self, epochNum):
        batchSize = 100
        loopCount = int(len(self._trainingData) / batchSize)
        for i in range(0, epochNum):
            for j in range(0, loopCount):
                #if j % 10 == 0:
                #    print(f"batch number {j}")
                self.feedForward(self._trainingData[j*batchSize:(j+1)*batchSize])
                self.backpropagate(self._trainingLabels[j*batchSize:(j+1)*batchSize])   
            print(f"epoch number {i} completed")
            if i % 100 == 0:
                print(f"for epoch number {i}:")
                self.testAccuracy()



    def testAccuracy(self):
        correctCount = 0
        """
        for i in range(0, 1000):
            print(self._trainingData[10000 + i])
            print(self._trainingLabels[10000 + i])
            print("Expected answer")
            #maximum value in the output array
            print(np.argmax(self._trainingLabels[10000 + i], axis=0))
            output = self.feedForward([self._trainingData[10000 + i]])
            print("Actual answer")
            print(np.argmax(output, axis=0))
            #fix
            print(np.argmax(output, axis=0), np.argmax(self._trainingLabels[10000 + i], axis=0))
            if np.argmax(output, axis=0) == np.argmax(self._trainingLabels[10000 + i], axis=0):
                correctCount += 1
                print(correctCount)
        """
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
            #if answerKey[idx] != np.argmax(self._testingLabels[idx]):
            showList.append(idx)
            idx += 1


        print("%%")
        print(showList)

        f, subpl = plt.subplots(5, 10)
        for i in range(0, 5):
            for j in range(0, 10):
                subpl[i][j].imshow(self._testingData[i * 10 + j].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
                currentConfidence = f"{(confidences[i * 10 + j]):.{3}f}"
                subpl[i][j].set_title(reverseDict[answerKey[i * 10 + j]] + "\n" + currentConfidence)
                subpl[i][j].axis('off')
        plt.show()


    def saveNetwork(self, path):
        print("Saving network...")

        #going to flatten arrays to save them, and then reshape them afterwards using shapes i'll store

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

            
    def loadNetwork(self, path):
        #setting the weights
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
