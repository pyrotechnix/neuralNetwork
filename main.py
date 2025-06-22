import neural_network
import numpy as np
import idx2numpy
from matplotlib import pyplot as plt


data_dir = "mnist_data"

#TO DO: Fix shape of batches so that the loss is actually correct
#FIXED
#backpropogation
#POSSIBLY DONE I DONT KNOW
#Fix softmax so that it doesn't have zeros theres an explaination just follow it
#FIXED
#if I can be fucked fix the matrixes so that it doesn't use reduntant functions
#FIXED
#Impliment testing
#FIXED
#Impliment saving of network so that I dont have to retrain every time

#Impliment ui (probably with tkinter)
#impliment installer

#CHANGES
#Add softmax instead of sigmoid for the output layer, since its better for class classification
#switched to using multi class cross entropy outputs
#switched to ReLU since sigmoid has issues
#Changed softmax function to subtract a value from all exponentials to avoid over/underflow


def main():
    np.set_printoptions(suppress=False, precision=3)
    #copied from gemini:
    train_images_path = 'mnist_data/train-images.idx3-ubyte'
    train_labels_path = 'mnist_data/train-labels.idx1-ubyte'
    test_images_path = 'mnist_data/t10k-images.idx3-ubyte'
    test_labels_path = 'mnist_data/t10k-labels.idx1-ubyte'
    trainingData = idx2numpy.convert_from_file(train_images_path)
    trainingLabels = idx2numpy.convert_from_file(train_labels_path)
    testingData = idx2numpy.convert_from_file(test_images_path)
    testingLabels = idx2numpy.convert_from_file(test_labels_path)
    #784 input layer 2x layers of 16 10 output neurons corresponding to each letter
    nn = neural_network.neuralNetwork([784, 100, 10])
    #nn.setValues([[[4, 8], [1, 10]], [[1, 1]]], [[[1], [4]], [[0], [0]]])
    #
    arr = trainingData[0]
    nn.setTrainingData(trainingData, trainingLabels, testingData, testingLabels)
    while True:
        print("Please select an action:")
        print("\tT: Train network")
        print("\tS: Save network")
        print("\tL: Load network")
        print("\tR: Test network")
        print("\tQ: Quit")
        trainInput = input("Action: ")
        if trainInput == 't' or trainInput == 'T':
            epochCount = int(input("How many epochs? "))
            print("Training...")
            nn.train(epochCount)
            nn.testAccuracy()
        elif trainInput == 's' or trainInput == 'S':
            nn.saveNetwork("neuralNetwork")
        elif trainInput == 'l' or trainInput == 'L':
            nn.loadNetwork("neuralNetwork")
        elif trainInput == 'r' or trainInput == 'R':
            nn.displayAnswers()
            nn.testAccuracy()
        elif trainInput == 'q' or trainInput == 'Q':
            break
        else:
            print("Invalid input, please retry")


"""
    f, subpl = plt.subplots(5, 10)
    for i in range(0, 5):
        for j in range(0, 10):
            subpl[i][j].imshow(testingData[i * 10 + j], cmap='gray', vmin=0, vmax=255)
            subpl[i][j].set_title(testingLabels[i * 10 + j])
            subpl[i][j].axis('off')
    plt.show()
"""



main()





