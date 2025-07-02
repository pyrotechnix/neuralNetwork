## REQUIREMENTS
- Numpy
  - Install with 'pip install numpy'
- MatPlotLib
  - Install with 'pip install matplotlib'
- PIL
  - Install with 'pip install pillow'

## FONTS USED IN TRAINING
For a full list of fonts used, please see font_list.txt

## USAGE
Once these libraries are installed, the program is ready to use. To use the program, run main.py. The program will automatically load the pre-trained neural network. To train it yourself, specify training data, and then use the train function to train it for any number of epochs. You can also save your network after training it, but please note that this will overwrite the pre-trained one. To make an input, select the "Make input" option, and then select a file directory. The network will then go through that folder and make predictions for each image in that folder. The program will display the fist 40 results using matplot lib, as well as displaying predictions for all files by file name.
