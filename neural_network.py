import numpy as np
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

    inputCount = None
    # defining variables to help keep track of the number of inputs and outputs, these are just defined with
    # layerInformation anyway, so it's not really that important
    _weightMatrix = None
    _biasMatrix = None
    _trainingData = None
    _trainingLabels = None
    _testingData = None
    _testingLabels = None
    _currentInput = None
    _currentOutput = None
    # Stored activation values across the entire network when data is fed through
    _activationValues = None
    # Dictionary keeping track of which catagory responds to which output neuron
    _catagoryDict = None
    _reverseDict = None
    # List of catagories
    _categories = None

    # takes layerInformation as list of ints detailing how many neurons per layer
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

            # creating an array in the correct shape to later store activation values
            for i in range(0, len(layerInformation)):
                layer = np.zeros(layerInformation[i])
                self._activationValues.append(layer.reshape(-1, 1))


    def reset(self):
        print("Resetting network...")
        # initialising random weights
        for i in range(1, len(self._weightMatrix)):
            originalShape = self._weightMatrix[i].shape
            weightLayer = np.random.randn(*originalShape) / 10
            self._weightMatrix[i] = weightLayer

        # initialising random biases
        for i in range(1, len(self._biasMatrix)):
            originalShape = self._biasMatrix[i].shape
            biasLayer = np.random.randn(*originalShape) / 10
            self._biasMatrix[i] = biasLayer
        print("Network reset")


    # trainingData should be only a 2-dimensional array, as it is converted into a 2-dimensional matrix later
    def setTrainingData(self, trainingData, trainingLabels, testingData, testingLabels):
        self._trainingData = np.zeros([len(trainingData), len(trainingData[0].flatten())])
        self._testingData = np.zeros([len(testingData), len(trainingData[0].flatten())])
        for i in range(0, len(self._trainingData)):
            self._trainingData[i] = trainingData[i].flatten()
        for i in range(0, len(self._testingData)):
            self._testingData[i] = testingData[i].flatten()
        # diving all values by the max to keep them in between 0 and 1 instead of making them big, since the
        # numpy exp function doesn't like large numbers, and its considered good practice anyway
        # Normalizing I guess you could call it
        self._trainingData /= self._trainingData.max()
        self._testingData /= self._testingData.max()
        self.interpretcategories(trainingLabels)
        self._trainingLabels = np.zeros([len(trainingData), len(self._catagoryDict), 1])
        # Assigning each label to its corresponding output neuron
        for i in range(0, len(trainingLabels)):
            self._trainingLabels[i] = self._catagoryDict[str(trainingLabels[i])]
        self._testingLabels = np.zeros([len(testingData), len(self._catagoryDict), 1])
        for i in range(0, len(testingLabels)):
            self._testingLabels[i] = self._catagoryDict[str(testingLabels[i])]

    # converts labels to a list of categories, and assigns an output neuron for each
    # this assumes that the number of output neurons is already correct
    def interpretcategories(self, labels):
        self._categories = []
        for i in labels:
            if i not in self._categories:
                self._categories.append(i)
        categoryCount = len(self._categories)
        for i in range(0, len(self._categories)):
            self._categories[i] = str(self._categories[i])
        self._catagoryDict = {}
        for i in range(0, categoryCount):
            addArr = np.array([np.zeros(categoryCount)])
            addArr[0][i] = 1
            self._catagoryDict[self._categories[i]] = addArr.T
        self._reverseDict = {}
        for i in self._catagoryDict:
            self._reverseDict[np.argmax(self._catagoryDict[i])] = i
        return self._catagoryDict

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

    # backpropagates through the network using gradient descent, takes expected but to be honest
    # in 99% of cases expected will just be training labels. However, it helps to pass it as a
    # paramater due to the batch training approach. Since current outputs are stored, it is
    # easier to simply pass the labels corresponding to those outputs, rather than to try and work
    # out where they are in the label array
    # note: DOES NOT FEED FORWARD! This function assumes that the feed forward has already happened!
    def backpropagate(self, expected):
        learningRate = 0.0005
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
            # traverses change array backwads, since I append to them in reverese order.
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
                # handles cases where the number of samples is not divisible by the batch size
                self.feedForward((self._trainingData[-(len(self._trainingData) % batchSize):]))
                self.backpropagate((self._trainingLabels[-(len(self._trainingLabels) % batchSize):]))
            print(f"epoch number {i} completed")
            if i % 10 == 0:
                print(f"for epoch number {i}:")
                self.testAccuracy()

    # Testing the accuracy of the network
    # very simple, feeds forward training and testing data and assesses accuracy.
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

    # Higher level function to make an input, since I didn't actually have one before
    def makeInput(self, arr, fileNames):
        self._currentInput = arr
        confidences = self.feedForward(arr)
        answerKey = np.argmax(self._activationValues[-1], axis=0)
        for i in range(0, len(arr)):
            print(f"Prediction for file {fileNames[i]}: {chr(int(self._reverseDict[answerKey[i]]))}")
            print(f"Confidence: {confidences[i]}")
            print("")

    # Displays answers using matplotlib
    # If toChar is true, it will assume that the inputs are ascii codes to convert to characters.
    def displayOutput(self, toChar):
        confidences = self.feedForward(self._currentInput)
        answerKey = np.argmax(self._activationValues[-1], axis=0)
        f, subpl = plt.subplots(4, 10)
        for i in range(0, 4):
            for j in range(0, 10):
                if (i * 10 + j) < len(confidences):
                    subpl[i][j].imshow(self._currentInput[i * 10 + j].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
                    currentConfidence = f"{(confidences[i * 10 + j]):.{3}f}"
                    if toChar:
                        subpl[i][j].set_title(chr(int(self._reverseDict[answerKey[i * 10 + j]])) + "\n" + currentConfidence)
                    else:
                        subpl[i][j].set_title(self._reverseDict[answerKey[i * 10 + j]] + "\n" + currentConfidence)
                    subpl[i][j].axis('off')
                else:
                    subpl[i][j].imshow(np.ones((28, 28)), cmap='gray', vmin=0, vmax=1)
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
        print("saving categories...")
        with open(path + "_categories", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self._categories)

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
        print("Loading categories...")
        with open(path + "_categories", 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            categories = None
            for row in reader:
                categories = np.array(row).astype(str)
            self.interpretcategories(categories)
        print('Loaded categories')
        print("Loaded!")


    def generateTestReport(self, inputs, fileNames):
        self._currentInput = inputs
        confidences = self.feedForward(inputs)
        answerKey = np.argmax(self._activationValues[-1], axis=0)
        actual = []
        fonts = []
        for i in range(0, len(fileNames)):
            actual.append(chr(int(fileNames[i].split('_')[-1].split('.')[0])))
            fonts.append(fileNames[i].split('_')[0])
        print(actual)
        passCount = 0
        failCount = 0
        with open('testReport.csv', 'w') as file:
            file.write("ID, Font, Prediction, Actual, Confidence, Pass/Fail\n")
            for i in range(0, len(answerKey)):
                addStr = f"{i}, "
                addStr += f"{fonts[i]}, "
                addStr += f"{chr(int(self._reverseDict[answerKey[i]]))}, "
                addStr += f"{actual[i]}, "
                addStr += f"{confidences[i]:.{3}f}, "
                if chr(int(self._reverseDict[answerKey[i]])) == actual[i]:
                    addStr += "PASS"
                    passCount += 1
                else:
                    addStr += "FAIL"
                    failCount += 1
                file.write(addStr + '\n')
            file.write(f"{passCount} PASSED {failCount} FAILED\n")
            print(f"TEST COMPLETED: {passCount} PASSED, {failCount} FAILED\n")
