import neural_network
import numpy as np
import idx2numpy
from matplotlib import pyplot as plt
import os
from PIL import Image

data_dir = "mnist_data"

#TO DO: Fix shape of batches so that the loss is actually correct
#FIXED
#backpropogation
#POSSIBLY DONE I DON'T KNOW -------- DONE
#Fix softmax so that it doesn't have zeros there's an explaination just follow it
#FIXED
#if I can be fucked fix the matrixes so that it doesn't use reduntant functions
#FIXED
#Impliment testing
#FIXED
#Impliment saving of network so that I don't have to retrain every time
#DONE

#Impliment ui (probably with tkinter)
#MISUNDERSTOOD You don't have to do this I might if I have time
#impliment installer
#MISUNDERSTOOD all I need to do is explain what libraries / python library is required

#CHANGES
#Add softmax instead of sigmoid for the output layer, since its better for class classification
#switched to using multi class cross entropy outputs
#switched to ReLU since sigmoid has issues
#Changed softmax function to subtract a value from all exponentials to avoid over/underflow


def loadImagesAndLabels(path):
    """
    Loads images and their corresponding labels from a specified directory.
    Images are expected to be 28x28. Labels are extracted from the image filenames.

    Args:
        path (str): The directory path containing the image files.

    Returns:
        tuple: A tuple containing:
            - imgArr (list of np.array): A list of images as NumPy arrays.
            - labelArr (list of str): A list of labels as strings.
    """
    imageNames = []
    imgArr = []
    labelArr = []
    print("Loading directories...")
    # List all entries in the given path
    for entry in os.listdir(path):
        # Filter for image files (assuming .png, adjust if different)
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
    nn = neural_network.neuralNetwork([784, 100, 50, 48])
    trainingData, trainingLabels = loadImagesAndLabels("trainingImages")
    testingData, testingLabels = loadImagesAndLabels("testingImages")
    print(trainingLabels[0])
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





