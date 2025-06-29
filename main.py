import neural_network
import numpy as np
import idx2numpy
import os
from PIL import Image

np.set_printoptions(suppress=False, precision=3)

# TO DO: Fix shape of batches so that the loss is actually correct
# FIXED
# backpropogation
# POSSIBLY DONE I DON'T KNOW -------- DONE
# Fix softmax so that it doesn't have zeros there's an explaination just follow it
# FIXED
# if I can be fucked fix the matrixes so that it doesn't use reduntant functions
# FIXED
# Impliment testing
# FIXED
# Impliment saving of network so that I don't have to retrain every time
# DONE

# Impliment ui (probably with tkinter)
# MISUNDERSTOOD You don't have to do this I might if I have time
# impliment installer
# MISUNDERSTOOD all I need to do is explain what libraries / python library is required

# CHANGES
# Add softmax instead of sigmoid for the output layer, since its better for class classification
# switched to using multi class cross entropy outputs
# switched to ReLU since sigmoid has issues
# Changed softmax function to subtract a value from all exponentials to avoid over/underflow
# Now using letters instead of mnist


def loadImagesAndLabels(path):
    imageNames = []
    imgArr = []
    labelArr = []
    print("Loading directories...")
    for entry in os.listdir(path):
        if entry.endswith('.png'):
            imageNames.append(entry)
    print(f"Found {len(imageNames)} images. Converting images....")

    for i in imageNames:
        # Open each image using PIL
        img = Image.open(os.path.join(path, i))
        # Convert image to NumPy array and append
        imgArr.append(np.array(img))
        # Extract label: takes the last element after splitting by '_'
        # and then removes the '.png' extension.
        labelArr.append(i.split('_')[-1].split('.')[0])
    print("Done!")
    return np.array(imgArr), np.array(labelArr)


def main():
    # If you want to use MNIST (Commenting out for now since I'm changing the defenitions later anyway

    """
    train_images_path = 'mnist_data/train-images.idx3-ubyte'
    train_labels_path = 'mnist_data/train-labels.idx1-ubyte'
    test_images_path = 'mnist_data/t10k-images.idx3-ubyte'
    test_labels_path = 'mnist_data/t10k-labels.idx1-ubyte'
    trainingData = idx2numpy.convert_from_file(train_images_path)
    trainingLabels = idx2numpy.convert_from_file(train_labels_path)
    testingData = idx2numpy.convert_from_file(test_images_path)
    testingLabels = idx2numpy.convert_from_file(test_labels_path)
    """

    nn = neural_network.neuralNetwork([784, 100, 50, 48])
    trainingData, trainingLabels = loadImagesAndLabels("trainingImages")
    testingData, testingLabels = loadImagesAndLabels("testingImages")
    nn.setTrainingData(trainingData, trainingLabels, testingData, testingLabels)


    running = True
    while running:
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
            running = False
        else:
            print("Invalid input, please retry")


main()
