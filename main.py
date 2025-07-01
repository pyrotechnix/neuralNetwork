import neural_network
import numpy as np
import os
import PIL
from PIL import Image
import random

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
# Fix output display so that if there are less than 40 images it won't break


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
    for j in range(-5, 5):
        for i in imageNames:
            # Open each image using PIL
            img = Image.open(os.path.join(path, i))
            # rotate image random amount
            img = img.rotate(float(4 * j), PIL.Image.NEAREST, expand=0)
            # Convert image to NumPy array and append
            imgArr.append(np.array(img))
            # Extract label: takes the last element after splitting by '_'
            # and then removes the '.png' extension.
            labelArr.append(i.split('_')[-1].split('.')[0])
    print("Done!")
    return np.array(imgArr), np.array(labelArr)


def loadImagesAndLabelsRandomRotation(path):
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
        img = img.convert('L')
        # rotate image random amount
        img = img.rotate(random.randint(-20, 20), PIL.Image.NEAREST, expand=0)
        # Convert image to NumPy array and append
        imgArr.append(np.array(img))
        # Extract label: takes the last element after splitting by '_'
        # and then removes the '.png' extension.
        labelArr.append(i.split('_')[-1].split('.')[0])
    print("Done!")
    return np.array(imgArr), np.array(labelArr)


def loadImages(path):
    imageNames = []
    imgArr = []

    print("Loading directories...")
    for entry in os.listdir(path):
        if entry.endswith('.png'):
            imageNames.append(entry)
    print(f"Found {len(imageNames)} images. Converting images....")
    for i in imageNames:
        # Open each image using PIL
        img = Image.open(os.path.join(path, i))
        img = img.convert('L')
        #img = img.rotate(random.randint(-20, 20), PIL.Image.NEAREST, expand=0)
        # Convert image to NumPy array and append
        imgArr.append(np.array(img))
    print("Done!")
    return np.array(imgArr), imageNames


def main():
    # If you want to use MNIST (Commenting out for now since I'm changing the defenitions later anyway

    """
    train_images_path = 'mnist_data/train-images.idx3-ubyte'
    train_labels_path = 'mnist_data/train-labels.idx1-ubyte'
    test_images_path = 'mnist_data/t10k-images.idx3-ubyte'
    test_labels_path = 'mnist_data/t10k-labels.idx1-ubyte'
    trainingData = idx2numpy.convert_from_file(train_images_path)
    trainingLabels = idx2numpy.convert_from_file(train_labels_path)lt
    testingData = idx2numpy.convert_from_file(test_images_path)
    testingLabels = idx2numpy.convert_from_file(testt_labels_path)
    """

    nn = neural_network.neuralNetwork([784, 100, 50, 74])

    # If you want the training data to load by default
    # Commented out because user will not be expected to have training or testing data

    """
    print("Loading default training data...")
    trainingData, trainingLabels = loadImagesAndLabels("character_images")
    print("Loading default testing data...")
    testingData, testingLabels = loadImagesAndLabelsRandomRotation("character_images")
    nn.setTrainingData(trainingData, trainingLabels, testingData, testingLabels)
    """

    nn.loadNetwork("neuralNetwork")

    running = True
    while running:
        print("Please select an action:")
        print("\tI: Load training/testing data")
        print("\tT: Train network")
        print("\tS: Save network")
        print("\tL: Load network")
        print("\tR: Make input")
        print("\tQ: Quit")
        trainInput = input("Action: ")

        if trainInput == 'i' or trainInput == 'I':
            print("Please ensure that all images end with an underscore followed by the character "
                  "(i.e. character_a.png)")
            print("faliure to do so may result in errors or unexpected results")
            directoryInput = input("Please enter directory: ")
            try:
                trainingData, trainingLabels = loadImagesAndLabels(directoryInput)
                testingData, testingLabels = loadImagesAndLabelsRandomRotation(directoryInput)
                nn.setTrainingData(trainingData, trainingLabels, testingData, testingLabels)
            except FileNotFoundError:
                print("Invalid file directory, please try again")

        elif trainInput == 't' or trainInput == 'T':
            epochCount = input("How many epochs? ")
            try:
                epochCount = int(epochCount)
                print("Training...")
                nn.train(epochCount)
                nn.testAccuracy()
            except ValueError:
                print("Please enter a valid integer")
            except TypeError:
                print("!-----------------------------------------------!\n")
                print("Please ensure that training data is loaded")
                print("\n!-----------------------------------------------!")
        elif trainInput == 's' or trainInput == 'S':
            nn.saveNetwork("neuralNetwork")

        elif trainInput == 'l' or trainInput == 'L':
            nn.loadNetwork("neuralNetwork")

        elif trainInput == 'r' or trainInput == 'R':
            directoryInput = input("Please enter directory: ")
            try:
                # Having to do a slight workaround, since I need the filenames to be passed in, but
                # I don't usually store them. But in this case, they are useful to identify output
                # if there are too many files to be displayed.
                newInput, inputFileNames = loadImages(directoryInput)
                nn.makeInput(newInput.reshape(len(newInput), -1) / 255, inputFileNames)
                nn.displayOutput(True)
                try:
                    nn.testAccuracy()
                except TypeError:
                    print("No training/testing data loaded")
            except FileNotFoundError:
                print("Invalid file directory, please try again")
            except TypeError:
                print("!-----------------------------------------------!\n")
                print("Please ensure that the neural network loaded is designed for this data")
                print("\n!-----------------------------------------------!")

        elif trainInput == 'q' or trainInput == 'Q':
            running = False

        else:
            print("Invalid input, please retry")


main()
