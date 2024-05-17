from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets, QtCore
import numpy as np


class MNISTLoaderNode(NodeBase):
    def __init__(self, name):
        super(MNISTLoaderNode, self).__init__(name)

        # Define the input pins
        self.num_classes = self.createInputPin('num_classes', 'IntPin')
        self.normalize = self.createInputPin('normalize', 'BoolPin')

        self.x_train = self.createOutputPin("x_train", 'AnyPin')
        self.y_train = self.createOutputPin("y_train", 'AnyPin')
        self.x_test = self.createOutputPin("x_test", 'AnyPin')
        self.y_test = self.createOutputPin("y_test", 'AnyPin')

        # Enable the allowAny option for the output pins
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.y_train.enableOptions(PinOptions.AllowAny)
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.y_test.enableOptions(PinOptions.AllowAny)

        self.get_started = self.createOutputPin("get_started", 'AnyPin')
        self.get_started.enableOptions(PinOptions.AllowAny)
        self.get_started.setData("No dataset loaded")

        self.dataset_description = self.createOutputPin("dataset_description", 'AnyPin')
        self.dataset_description.enableOptions(PinOptions.AllowAny)
        self.dataset_description.setData("No dataset loaded")

        self.dataset_dimension = self.createOutputPin("dataset_dimension", 'AnyPin')
        self.dataset_dimension.enableOptions(PinOptions.AllowAny)
        self.dataset_dimension.setData("No dataset loaded")

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('IntPin')
        helper.addInputDataType('BoolPin')
        helper.addOutputDataType('AnyPin')
        return helper

    def compute(self, *args, **kwargs):
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Set the data for the output pins
        self.x_train.setData(x_train)
        self.y_train.setData(y_train)
        self.x_test.setData(x_test)
        self.y_test.setData(y_test)

        # print(x_train)
        # print(y_train)
        # print(x_test)
        # print(y_test)

        # Update the number of classes in the custom layout
        num_classes = self.num_classes.getData()
        nr_classes = f"Classes (default is 10): {num_classes}"
        print(nr_classes)


        # Dataset description
        dataset_description = f"The MNIST dataset is a popular dataset for handwritten digit classification. It consists of grayscale images of handwritten digits from 0 to 9. Therefore, the MNIST dataset has 10 classes, each corresponding to one of the digits from 0 to 9. The classes in the MNIST dataset are as follows: Class 0: Digit 0 .. up to Class 9: Digit 9. When performing classification tasks using the MNIST dataset, the goal is to train a model that can accurately predict the correct digit class given an input image of a handwritten digit. The dataset consists of two main components: 1) Training set, that contains 60,000 images with corresponding labels, used for training machine learning models. 2) Test set, that contains 10,000 images with corresponding labels, used for evaluating the performance of trained models."
        self.dataset_description.setData(dataset_description)
        #print(dataset_description)

        # Update the dataset information in the custom layout
        dataset_dimension = f" Train data shape is {x_train.shape}, Train set labels shape is {y_train.shape}, Test set shape is {x_test.shape} and Test set labels shape is {y_test.shape}. \n \n \n Reminder:  the feature space in the MNIST dataset is defined by the 28x28 grid of pixel values for each image. The 'x_train' (i.e. training set images) has a shape of (60000, 28, 28) which means it consists of 60,000 images, each with dimensions 28x28 pixels. The 'y_train' (i.e. training set labels) has a shape of (60000,) which means it consists of 60,000 labels corresponding to the training images. The 'x_test' (i.e., test set images) has a shape of (10000, 28, 28) which means it consists of 10,000 images, each with dimensions 28x28 pixels. The 'y_test' (i.e., test set labels) has a shape of (10000,) which means it consists of 10,000 labels corresponding to the test images.\nSo, 'x_train' and 'x_test' are 3-dimensional arrays representing images, while 'y_train' and 'y_test' are 1-dimensional arrays representing labels or class values."
        self.dataset_dimension.setData(dataset_dimension)
        #print(dataset_dimension)

        # Get started
        get_started = f"Please, view 'dataset_description' and 'dataset_dimension'. Then, set 'num_classes' equal to 10. The size of the test set is already fixed at 14.5% (about 10,000) and so, 85.5% for training (about 60,000). Please, flag Normalize. \n \n Normalization: \n in the context of image data, the pixel values typically range from 0 to 255, representing the intensity of the pixel. In the MNIST dataset, the pixel values of the grayscale images also fall within this range. When you flag 'normalize', you divide the pixel values by 255. This is a common technique used to normalize the image data. Normalization is performed to scale the pixel values to a standard range, typically between 0 and 1 or -1 and 1. This process helps in improving the training stability and convergence of machine learning models. By dividing the pixel values by 255, each pixel value is transformed to a decimal value between 0 and 1. This normalization ensures that the pixel values are within a consistent range, making it easier for the model to learn patterns and make accurate predictions. It also helps in preventing issues such as numerical instability or dominance of certain pixel ranges over others during training. In the case of the MNIST dataset, dividing the pixel values by 255 ensures that the normalized pixel values range from 0 to 1, where 0 represents black (no intensity) and 1 represents white (maximum intensity). This normalization allows the model to effectively learn from the image data and make predictions based on the relative intensities of the pixels."
        self.get_started.setData(get_started)
        #print(get_started)

    def createUi(self):
        # Create the main widget for the custom UI
        widget = QtWidgets.QWidget()

        # Create a layout for the widget
        layout = QtWidgets.QVBoxLayout(widget)

        # Create labels to display the pin names and values
        pin_labels = [
            ("x_train", self.x_train),
            ("y_train", self.y_train),
            ("x_test", self.x_test),
            ("y_test", self.y_test),
            ("get_started", self.get_started),
            ("dataset_dimension", self.dataset_info),
            ("dataset_description", self.dataset_description),
            ("num_classes", self.num_classes),
            ("normalize", self.normalize),
        ]

        for pin_name, pin in pin_labels:
            label = QtWidgets.QLabel(pin_name)
            layout.addWidget(label)

            value_label = QtWidgets.QLabel()
            layout.addWidget(value_label)

            # Create a function to update the value label when the pin data changes
            def update_value_label(data):
                value_label.setText(str(data))

            pin.onPinDataChanged.connect(update_value_label)

        # Set the widget as the custom UI for the node
        self.setWidget(widget)

    @staticmethod
    def category():
        return '1_Data_Load'
