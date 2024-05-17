from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets, QtCore
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.datasets import reuters

class REUTLoaderNode(NodeBase):
    def __init__(self, name):
        super(REUTLoaderNode, self).__init__(name)

        # Define the input and output pins
        self.num_classes = self.createInputPin('num_classes', 'IntPin')
        self.max_words = self.createInputPin('max_words', 'IntPin')
        self.max_length = self.createInputPin('max_length', 'IntPin')
        self.test_size = self.createInputPin('test_size', 'FloatPin')

        self.x_train = self.createOutputPin("x_train", 'StringPin')
        self.y_train = self.createOutputPin("y_train", 'AnyPin')
        self.x_test = self.createOutputPin("x_test", 'StringPin')
        self.y_test = self.createOutputPin("y_test", 'AnyPin')
        #self.token_count = self.createOutputPin("token_count", 'IntPin')

        # Enable the allowAny option for the output pins
        #self.x_train.enableOptions(PinOptions.AllowAny)
        self.y_train.enableOptions(PinOptions.AllowAny)
        #self.x_test.enableOptions(PinOptions.AllowAny)
        self.y_test.enableOptions(PinOptions.AllowAny)
        #self.token_count.enableOptions(PinOptions.AllowAny)

        self.get_started = self.createOutputPin("get_started", 'AnyPin')
        self.get_started.enableOptions(PinOptions.AllowAny)
        self.get_started.setData("No dataset loaded")

        self.dataset_description = self.createOutputPin("dataset_description", 'AnyPin')
        self.dataset_description.enableOptions(PinOptions.AllowAny)
        self.dataset_description.setData("No dataset loaded")

        self.dataset_dimension = self.createOutputPin("dataset_dimension", 'AnyPin')
        self.dataset_dimension.enableOptions(PinOptions.AllowAny)
        self.dataset_dimension.setData("No dataset loaded")

        # Set default data for output pins
        #self.y_train.setData(None)
        #self.y_test.setData(None)
        #self.token_count.setData(0)

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('IntPin')
        helper.addOutputDataType('AnyPin')
        helper.addOutputDataType('StringPin')
        helper.addOutputDataType('FloatPin')
        helper.addOutputDataType('IntPin')
        return helper

    def compute(self, *args, **kwargs):
        # Load the Reuters newswire dataset
        max_words = self.max_words.getData()
        max_length = self.max_length.getData()
        test_size = self.test_size.getData()
        (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words)

        # Pad sequences to a fixed length
        x_train = pad_sequences(x_train, maxlen=max_length)
        x_test = pad_sequences(x_test, maxlen=max_length)

        # Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=42)

        # Set the data for the output pins
        self.x_train.setData(x_train)
        self.y_train.setData(y_train)
        self.x_test.setData(x_test)
        self.y_test.setData(y_test)

        # Update the number of classes and test size in the custom layout
        num_classes = self.num_classes.getData()
        test_size = self.test_size.getData()
        nr_classes_testsize = f"Number of classes (default is 2): {num_classes}\nTest set size: {test_size * 100}%\nTrain set size: {100-test_size * 100}%"
        print(nr_classes_testsize)
        max_words2 = f"Max_words is: {max_words}"
        print(max_words2)
        max_length2 = f"Max_length is: {max_length}"
        print(max_length2)

        # Compute the total number of tokens
        total_tokens_train = np.sum([len(sequence) for sequence in x_train])
        total_tokens_test = np.sum([len(sequence) for sequence in x_test])
        total_tokens_dataset = total_tokens_train + total_tokens_test

        total_tokens_train2 = f"Total_tokens in train set is: {total_tokens_train}"
        print(total_tokens_train2)
        total_tokens_test2 = f"Total_tokens in test set is: {total_tokens_test}"
        print(total_tokens_test2)
        total_tokens_dataset2 = f"Total_tokens in dataset is: {total_tokens_dataset}"
        print(total_tokens_dataset2)

        # Dataset description
        dataset_description = f"The Reuters dataset contains articles classified into different topics. It consists of a set of short newswire stories and their corresponding topics. It collects 11,228 samples and 46 topics (each label is an integer value ranging from 0 to 45 class). Each news article is represented as a list of word indexes, where each index corresponds to a word in the Reuters vocabulary. The dataset also includes a word index dictionary that maps words to their corresponding indexes. The topics cover commodities (about 24), finance/economics (about 16), Housing/Construction (about 3) and Livestock/Agriculture (about 2). These groupings provide a rough categorization based on the topic names. However, some topics may overlap or cover multiple areas, and the categorization may not be exhaustive or exclusive."
        self.dataset_description.setData(dataset_description)
        #print(dataset_description)

        # Update the dataset information in the custom layout
        dataset_dimension = f" Train data shape is {x_train.shape}, Train set labels shape is {y_train.shape}, Test set shape is {x_test.shape} and Test set labels shape is {y_test.shape}. \n \n \n Reminder: when Train data shape is (7185, 50) , the shape indicates that the training data (x_train) has 7185 samples (rows) and each sample has a sequence length of 50 (columns/features). When the train set labels shape is (7185,) , the shape indicates that the training set labels (y_train) are a 1D array with 7185 elements, corresponding to the labels for the 7185 training samples. When the test set shape is (1797, 50) , the shape indicates that the test data (x_test) has 1797 samples (rows) and each sample has a sequence length of 50 (columns/features). Finally, when the test set labels shape is (1797,) , the shape indicates that the test set labels (y_test) are a 1D array with 1797 elements, corresponding to the labels for the 1797 test samples."
        self.dataset_dimension.setData(dataset_dimension)
        #print(dataset_dimension)

        # Get started
        get_started = f"Please, view 'dataset_description' and 'dataset_dimension'. Set 'num_classes' equal to 46. If you set 'max_length' equal to 100, then Train and test data shape is (.., 100)"
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
            ("num_classes", self.num_classes),
            ("get_started", self.get_started),
            ("dataset_dimension", self.dataset_info),
            ("dataset_description", self.dataset_description),
            #("token_count", self.token_count),
            ("max_words", self.max_words),
            ("max_length", self.max_length),
            ("test_size", self.test_size),
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
