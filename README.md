Simple MLP neural network currently being built to train on a dataset of 28*28 images.

Specifications:
- Using leaky ReLU activation
- Using softmax and multiclass cross entropy loss for last layer and cost function
- Using NumPY for calculations
- Currently has 1 hidden layer of 100 neurons
- Achieved 95% test accuracy on unseen data using MNIST dataset
- Achieved 95+% test accuracy on character recognition with a variety of fonts randomly rotated up to 20 degrees

Requirements:
- Numpy (For the bulk of the calculations, I am using numpy matrices)
- Idx2numpy (I am using this to convert the MNIST data to numpy arrays easily, I'm sure there are other ways but this way works well for me)
  - Note: This is no longer required unless you want to train the dataset on mnist instead of either using the pre-trained dataset or training it on your own images
- MatPlotLib (Currently I am using this to visualise the predictions once they have been made)
- PIL (Used to convert images into a format that the neural network can work with)
