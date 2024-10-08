from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets, QtCore
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
#from PyQt5.QtWidgets import QLabel, QInputDialog
from PySide2.QtWidgets import QLabel, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QPushButton
from PySide2.QtCore import Qt
from PyFlow.Core.Common import *
from sklearn import metrics
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from matplotlib.table import table
from collections import Counter

class MOVLoaderNode(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False

        ####################### prompt only on refreshing this node
        async def refresh_node(node):
            # Check if the node is of type 'MOVLoaderNode'
            if node.type == 'MOVLoaderNode':
                # Show the prompt for 'MOVLoaderNode'
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')

        #######################

        super(MOVLoaderNode, self).__init__(name)

        # Define the input and output pins
        self.num_classes = self.createInputPin("num_classes", 'IntPin')
        self.max_words = self.createInputPin("max_words", 'IntPin')
        #self.max_length = self.createInputPin("max_length", 'IntPin')
        self.test_size = self.createInputPin("test_size", 'FloatPin')

        self.x_train = self.createOutputPin("x_train", 'AnyPin')
        self.y_train = self.createOutputPin("y_train", 'AnyPin')
        self.x_test = self.createOutputPin("x_test", 'AnyPin')
        self.y_test = self.createOutputPin("y_test", 'AnyPin')
        self.dataset = self.createOutputPin("dataset", 'AnyPin')

        # Enable the allowAny option for the output pins
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.y_train.enableOptions(PinOptions.AllowAny)
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.y_test.enableOptions(PinOptions.AllowAny)
        self.dataset.enableOptions(PinOptions.AllowAny)

        self.dataset_description = self.createOutputPin("dataset_description", 'StringPin')
        self.dataset_dimension = self.createOutputPin("dataset_dimension", 'StringPin')
        self.tokenization = self.createOutputPin("tokenization", 'StringPin')
        self.parameters = self.createOutputPin("parameters", 'StringPin')
        self.examples = self.createOutputPin("examples", 'StringPin')
        self.counters = self.createOutputPin("counters", 'StringPin')

        # Set default values for prompt
        self.num_classes.setData(2)
        self.max_words.setData(100)
        #self.max_length.setData(50)
        self.test_size.setData(0.2)

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('IntPin')
        helper.addInputDataType('IntPin')  # num_classes
        helper.addInputDataType('IntPin')  # max_words
        #helper.addInputDataType('IntPin')  # max_length
        helper.addInputDataType('FloatPin')  # test_size
        helper.addOutputDataType('AnyPin')
        helper.addOutputDataType('StringPin')
        helper.addOutputDataType('FloatPin')
        helper.addOutputDataType('IntPin')
        return helper

    def promptVariables(self):
        # Create a dialog to confirm the refresh
        refresh_dialog = QtWidgets.QMessageBox()
        refresh_dialog.setWindowTitle("Settings")
        refresh_dialog.setText("Do you want to set parameters? If already done, choose No")
        refresh_dialog.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        refresh_dialog.setDefaultButton(QtWidgets.QMessageBox.No)

        # Display the dialog and check the user's response
        response = refresh_dialog.exec_()

        if response == QtWidgets.QMessageBox.Yes:

            # Prompt the user to set the values of a and b
            num_classes_default = self.num_classes.getData()
            max_words_default = self.max_words.getData()
            #max_length_default = self.max_length.getData()
            test_size_default = self.test_size.getData()

            # Customize the name and size of the prompt window
            window_name = "Set parameters"
            window_width = 600

            num_classes_dialog = QtWidgets.QInputDialog()
            num_classes_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)
            num_classes_dialog.setIntRange(-2147483648, 2147483647)
            num_classes_dialog.setWindowTitle(window_name)
            num_classes_dialog.setLabelText("Enter value for 'num_classes':")
            num_classes_dialog.setOkButtonText("OK")
            num_classes_dialog.setCancelButtonText("Cancel")

            max_words_dialog = QtWidgets.QInputDialog()
            max_words_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)
            max_words_dialog.setIntRange(-2147483648, 2147483647)
            max_words_dialog.setWindowTitle(window_name)
            max_words_dialog.setLabelText("Enter value for 'max_words':")
            max_words_dialog.setOkButtonText("OK")
            max_words_dialog.setCancelButtonText("Cancel")

            #max_length_dialog = QtWidgets.QInputDialog()
            #max_length_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)
            #max_length_dialog.setIntRange(-2147483648, 2147483647)
            #max_length_dialog.setWindowTitle(window_name)
            #max_length_dialog.setLabelText("Enter value for 'max_length':")
            #max_length_dialog.setOkButtonText("OK")
            #max_length_dialog.setCancelButtonText("Cancel")

            test_size_dialog = QtWidgets.QInputDialog()
            test_size_dialog.setInputMode(QtWidgets.QInputDialog.DoubleInput)
            test_size_dialog.setDoubleRange(0.0, 1.0)
            test_size_dialog.setWindowTitle(window_name)
            test_size_dialog.setLabelText("Enter value for 'test_size':")
            test_size_dialog.setOkButtonText("OK")
            test_size_dialog.setCancelButtonText("Cancel")

            # Set the width of the prompt windows
            num_classes_dialog.resize(window_width, num_classes_dialog.height())
            max_words_dialog.resize(window_width, max_words_dialog.height())
            #max_length_dialog.resize(window_width, max_length_dialog.height())
            test_size_dialog.resize(window_width, test_size_dialog.height())

            num_classes_dialog.setIntValue(num_classes_default)
            if num_classes_dialog.exec_() == QtWidgets.QDialog.Accepted:
                num_classes_value = num_classes_dialog.intValue()
            else:
                # User canceled, so show a message and exit the prompt
                num_classes_dialog.reject()  # Close the dialog
                return

            max_words_dialog.setIntValue(max_words_default)
            if max_words_dialog.exec_() == QtWidgets.QDialog.Accepted:
                max_words_value = max_words_dialog.intValue()
            else:
                # User canceled, so show a message and exit the prompt
                max_words_dialog.reject()  # Close the dialog
                return

            #max_length_dialog.setIntValue(max_length_default)
            #if max_length_dialog.exec_() == QtWidgets.QDialog.Accepted:
            #    max_length_value = max_length_dialog.intValue()
            #else:
                # User canceled, so show a message and exit the prompt
            #    max_length_dialog.reject()  # Close the dialog
            #    return

            test_size_dialog.setDoubleValue(test_size_default)
            if test_size_dialog.exec_() == QtWidgets.QDialog.Accepted:
                test_size_str = str(test_size_dialog.doubleValue())
                test_size_value = float(test_size_str)
            else:
                # User canceled, so show a message and exit the prompt
                test_size_dialog.reject()  # Close the dialog
                return

            # Set the data of pins with the entered values
            self.num_classes.setData(num_classes_value)
            self.max_words.setData(max_words_value)
            #self.max_length.setData(max_length_value)
            self.test_size.setData(test_size_value)

            # Refresh the data
            self.data_refreshed = True
            self.compute()
            return

        else:
            # User chose not to refresh the data
            self.num_classes.setData(2)
            self.max_words.setData(100)
            #self.max_length.setData(50)
            self.test_size.setData(0.2)
            return

    def showDataNotRefreshedMessage(self):
        message_box = QtWidgets.QMessageBox()
        message_box.setWindowTitle("Settings")
        message_box.setText("Default parameters have been set.")
        message_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        message_box.exec_()

    def refresh(self, pin):
        # Call the function to display the dataset description
        self.display_dataset_description()

    def compute(self, *args, **kwargs):
        # Load dataset
        (x_train, y_train), (x_test, y_test) = imdb.load_data()

        # Convert labels to a list
        labels = list(y_train) + list(y_test)

        # Retrieve the values of the input pins
        test_size = self.test_size.getData()
        num_classes = self.num_classes.getData()
        max_words = self.max_words.getData()
        # = self.max_length.getData()

        # Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=42)

        # Tokenize the text
        tokenizer = Tokenizer(num_words=max_words)

        # Pad sequences to a fixed length
        # x_train = pad_sequences(x_train, maxlen=max_length)
        # x_test = pad_sequences(x_test, maxlen=max_length)

        # Convert labels to one-hot encoding (assuming your labels are already integer-encoded)
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        # Concatenate the train and test data and labels
        texts = np.concatenate((x_train, x_test), axis=0)

        ######################### COUNT

        # Create a list of label classes
        label_classes = list(set(labels))

        # Count the occurrences of each label type in y_train
        y_train_labels = [label_classes[np.argmax(label)] for label in y_train]
        y_train_label_counts = Counter(y_train_labels)
        negative_count_train = y_train_label_counts['negative']
        positive_count_train = y_train_label_counts['positive']

        print("Number of negative reviews in y_train:", negative_count_train)
        print("Number of positive reviews in y_train:", positive_count_train)

        # Count the occurrences of each label type in y_test
        y_test_labels = [label_classes[np.argmax(label)] for label in y_test]
        y_test_label_counts = Counter(y_test_labels)
        negative_count_test = y_test_label_counts['negative']
        positive_count_test = y_test_label_counts['positive']

        print("Number of negative reviews in y_test:", negative_count_test)
        print("Number of positive reviews in y_test:", positive_count_test)

        # Update the number of classes, test size, max words, and lengths in the custom layout
        num_classes = self.num_classes.getData()
        test_size = self.test_size.getData()
        nr_classes_testsize = f"Number of classes (default is 2): {num_classes}\nTest set size: {test_size * 100}%\nTrain set size: {100 - test_size * 100}%"
        print(nr_classes_testsize)
        max_words2 = f"Max_words is: {max_words}"
        print(max_words2)
        # max_length2 = f"Max_length is: {max_length}"
        # print(max_length2)

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
        dataset_description = "The IMDB dataset contains a collection of movie reviews, and each review is labeled as either positive or negative. The dataset is commonly used for sentiment analysis tasks. The IMDB dataset contains a total of 50,000 movie reviews, with an equal number of positive and negative reviews. These reviews are split into a training set and a test set, with 25,000 reviews in each. Each review is represented as a sequence of integers, where each integer corresponds to a specific word in a predefined vocabulary. The dataset has already been preprocessed and encoded in this format, where words are indexed by their frequency of occurrence in the overall dataset."

        def convert_string_to_image(string):
            # Create a figure and plot the string as text
            plt.figure(figsize=(10, 4))
            plt.axis('off')
            plt.text(0.5, 0.5, string, fontsize=18, ha='center', va='center', wrap=True, fontname='DejaVu Sans')

            # Save the figure as a PNG file
            image_path = 'Dataset_description.png'
            plt.savefig(image_path)
            plt.close()

            return image_path

        image_path = convert_string_to_image(dataset_description)
        self.dataset_description.setData(image_path)

        # Dataset examples
        texts2 = texts[:5]

        def convert_strings_to_image(strings):
            # Create a figure with subplots for each string
            fig, axs = plt.subplots(len(strings), 1, figsize=(10, 6))

            # Iterate over the strings and plot them as text in each subplot
            for i, string in enumerate(strings):
                axs[i].axis('off')
                axs[i].text(0.01, 0.01, string, fontsize=18, ha='left', va='center', wrap=True, fontname='DejaVu Sans')

            # Adjust the spacing between subplots
            plt.subplots_adjust(hspace=2)

            # Save the figure as a PNG file
            image_path2 = 'Dataset_examples.png'
            plt.savefig(image_path2)
            plt.close()

            return image_path2

        image_path2 = convert_strings_to_image(texts2)
        self.examples.setData(image_path2)

        # Dataset Dimension Table
        v1 = x_train.shape
        v2 = y_train.shape
        v3 = x_test.shape
        v4 = y_test.shape

        dimension = pd.DataFrame({
            'Train data \n (n_samples, n_features)': [v1],
            'Train labels \n (n_samples, n_labels)': [v2],
            'Test data \n (n_samples, n_features)': [v3],
            'Test labels \n (n_samples, n_labels)': [v4]
        })

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(18, 4))

        # Create the table and adjust font size and cell padding
        table5 = plt.table(cellText=dimension.values, colLabels=dimension.columns, cellLoc='center', loc='center')

        # Adjust font size
        table5.set_fontsize(18)

        # Color the cells in the first row with light blue
        for cell in table5.get_celld().values():
            if cell.get_text().get_text() in dimension.columns:
                cell.set_facecolor('lightblue')

        # Enable content wrapping in cells
        for cell in table5.get_celld().values():
            cell.set_text_props(wrap=True)

        # Adjust cell size
        table5.scale(1, 9.5)  # Increase the cell size by a factor of 1.5

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('dataset_dimension.png')
        plt.close()

        self.dataset_dimension.setData('dataset_dimension.png')

        # Parameters

        parameter = pd.DataFrame({
            'Nr. Classes': [num_classes],
            'Train set \nsize': [f'{100 - test_size * 100}%'],
            'Test set \nsize': [f'{test_size * 100}%'],
            'Max words': [max_words]
            #'Max Length': [max_length]
        })

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(18, 4))

        # Create the table and adjust font size and cell padding
        table6 = plt.table(cellText=parameter.values, colLabels=parameter.columns, cellLoc='center', loc='center')

        # Adjust font size
        table6.set_fontsize(18)

        # Color the cells in the first row with light blue
        for cell in table6.get_celld().values():
            if cell.get_text().get_text() in parameter.columns:
                cell.set_facecolor('lightblue')

        # Enable content wrapping in cells
        for cell in table6.get_celld().values():
            cell.set_text_props(wrap=True)

        # Adjust cell size
        table6.scale(1, 6.5)  # Increase the cell size by a factor of 1.5

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('parameters.png')
        plt.close()

        self.parameters.setData('parameters.png')

        # Tokenization data

        # Count the number of tokens in each sequence
        num_tokens_train = [len(sequence) for sequence in x_train]
        num_tokens_test = [len(sequence) for sequence in x_test]

        # Compute the total number of tokens
        total_tokens_train = sum(num_tokens_train)
        total_tokens_test = sum(num_tokens_test)
        total_tokens_dataset = total_tokens_train + total_tokens_test

        # Print the token counts
        print("Total tokens in train set:", total_tokens_train)
        print("Total tokens in test set:", total_tokens_test)
        print("Total tokens in dataset:", total_tokens_dataset)

        tokenizat = pd.DataFrame({
            'Dataset Total \ntokens': [f'{total_tokens_dataset:,}'],
            'Train set \ntokens': [f'{total_tokens_train:,}'],
            'Test set \ntokens': [f'{total_tokens_test:,}']
        })

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(10, 8))

        # Create the table and adjust font size and cell padding
        table7 = plt.table(cellText=tokenizat.values, colLabels=tokenizat.columns, cellLoc='center', loc='center')

        # Adjust font size
        table7.set_fontsize(18)

        # Color the cells in the first row with light blue
        for cell in table7.get_celld().values():
            if cell.get_text().get_text() in tokenizat.columns:
                cell.set_facecolor('lightblue')

        # Enable content wrapping in cells
        for cell in table7.get_celld().values():
            cell.set_text_props(wrap=True)

        # Adjust cell size
        table7.scale(1, 6.5)  # Increase the cell size by a factor of 1.5

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('tokenization.png')
        plt.close()

        self.tokenization.setData('tokenization.png')

        # Counter

        # Concatenate the train and test labels to get labels for the entire dataset
        y_combined = np.concatenate((y_train, y_test), axis=0)

        # Count the occurrences of each label type for the entire dataset
        y_combined_labels = [label_classes[np.argmax(label)] for label in y_combined]
        y_combined_label_counts = Counter(y_combined_labels)
        negative_count = y_combined_label_counts['negative']
        positive_count = y_combined_label_counts['positive']

        # Get the unique labels and count their occurrences
        unique_labels = list(set(y_combined_labels))
        label_counts = Counter(y_combined_labels)

        counter = pd.DataFrame({
            'Tot. Negative': [negative_count],
            'Tot. Positive': [positive_count],
            'Tot. Negative\nin train': [negative_count_train],
            'Tot. Positive\nin train': [positive_count_train],
            'Tot. Negative\nin test': [negative_count_test],
            'Tot. Positive\nin test': [positive_count_test],
            'Classes': [unique_labels]
        })

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(18, 4))

        # Create the table and adjust font size and cell padding
        table8 = plt.table(cellText=counter.values, colLabels=counter.columns, cellLoc='center', loc='center')

        # Adjust font size
        table8.set_fontsize(18)

        # Color the cells in the first row with light blue
        for cell in table8.get_celld().values():
            if cell.get_text().get_text() in counter.columns:
                cell.set_facecolor('lightblue')

        # Enable content wrapping in cells
        for cell in table8.get_celld().values():
            cell.set_text_props(wrap=True)

        # Adjust cell size
        table8.scale(1, 8.5)  # Increase the cell size by a factor of 1.5

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('counters.png')
        plt.close()

        self.counters.setData('counters.png')

        #####################################################

        # Check if values are set, if not prompt the user
        if not self.num_classes.hasConnections() or not self.max_words.hasConnections() or not self.test_size.hasConnections():
            self.promptVariables()

        num_classes = self.num_classes.getData()
        max_words = self.max_words.getData()
        #max_length = self.max_length.getData()
        test_size = self.test_size.getData()

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
            ("dataset_dimension", self.dataset_dimension),
            ("dataset_description", self.dataset_description),
            ("tokenization", self.tokenization),
            ("parameters", self.parameters),
            ("num_classes", self.num_classes),
            ("max_words", self.max_words),
            #("max_length", self.max_length),
            ("test_size", self.test_size),
            ("examples", self.examples),
            ("dataset", self.dataset),
            ("counters", self.counters),
        ]

        for pin_name, pin_prompt in pin_labels:
            label = QtWidgets.QLabel(pin_name)
            layout.addWidget(label)

            value_label = QtWidgets.QLabel()
            layout.addWidget(value_label)

            # Create a function to update the value label when the pin data changes
            def update_value_label(data):
                value_label.setText(str(data))

            pin = getattr(self, pin_prompt.getName())
            pin.onPinDataChanged.connect(update_value_label)
            update_value_label(pin.getData())

        # Set the widget as the custom UI for the node
        self.setWidget(widget)


    @staticmethod
    def category():
        return '1_Data_Load'
