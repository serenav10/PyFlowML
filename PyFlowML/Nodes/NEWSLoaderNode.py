from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets, QtCore
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import time
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

# 20 Newsgroups dataset
class NEWSLoaderNode(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False

        ####################### prompt only on refreshing this node
        async def refresh_node(node):
            # Check if the node is of type 'NEWSLoaderNode'
            if node.type == 'NEWSLoaderNode':
                # Show the prompt for 'NEWSLoaderNode'
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')

        #######################

        super(NEWSLoaderNode, self).__init__(name)

        # Define the input and output pins
        self.num_classes = self.createInputPin("num_classes", 'IntPin')
        self.max_words = self.createInputPin("max_words", 'IntPin')
        self.max_length = self.createInputPin("max_length", 'IntPin')
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
        self.num_classes.setData(20)
        self.max_words.setData(100)
        self.max_length.setData(1000)
        self.test_size.setData(0.2)

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('IntPin')
        helper.addInputDataType('IntPin')  # num_classes
        helper.addInputDataType('IntPin')  # max_words
        helper.addInputDataType('IntPin')  # max_length
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
            max_length_default = self.max_length.getData()
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

            max_length_dialog = QtWidgets.QInputDialog()
            max_length_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)
            max_length_dialog.setIntRange(-2147483648, 2147483647)
            max_length_dialog.setWindowTitle(window_name)
            max_length_dialog.setLabelText("Enter value for 'max_length':")
            max_length_dialog.setOkButtonText("OK")
            max_length_dialog.setCancelButtonText("Cancel")

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
            max_length_dialog.resize(window_width, max_length_dialog.height())
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

            max_length_dialog.setIntValue(max_length_default)
            if max_length_dialog.exec_() == QtWidgets.QDialog.Accepted:
                max_length_value = max_length_dialog.intValue()
            else:
                # User canceled, so show a message and exit the prompt
                max_length_dialog.reject()  # Close the dialog
                return

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
            self.max_length.setData(max_length_value)
            self.test_size.setData(test_size_value)

            # Refresh the data
            self.data_refreshed = True
            self.compute()
            return

        else:
            # User chose not to refresh the data
            self.num_classes.setData(20)
            self.max_words.setData(100)
            self.max_length.setData(1000)
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
        # Load the 20 Newsgroups dataset
        num_classes = self.num_classes.getData()
        max_words = self.max_words.getData()
        max_length = self.max_length.getData()
        test_size = self.test_size.getData()

        categories = None  # Use all categories
        newsgroups_data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

        # Split the data into train and test sets
        #x_train, x_test, y_train, y_test = train_test_split(newsgroups_data.data, newsgroups_data.target, test_size=test_size, random_state=42)

        # Process the data to extract text and labels
        texts = newsgroups_data.data
        labels = [newsgroups_data.target_names[i] for i in newsgroups_data.target]

        ######################### COUNT

        # Print the first five elements of texts
        if len(texts) > 0:
            print(texts[:5])

        # Print the unique labels
        unique_labels = list(set(labels))
        print(unique_labels)

        # Count the occurrences of each label type
        label_counts = Counter(labels)
        print("Label counts:", label_counts)

        def map_label(label):
            if label.startswith('alt.atheism'):
                return 'atheism'
            elif label.startswith('comp.graphics'):
                return 'graphics'
            elif label.startswith('comp.os.ms-windows.misc'):
                return 'windows'
            elif label.startswith('comp.sys.ibm.pc.hardware'):
                return 'pc_hardware'
            elif label.startswith('comp.sys.mac.hardware'):
                return 'mac_hardware'
            elif label.startswith('comp.windows.x'):
                return 'windows_x'
            elif label.startswith('misc.forsale'):
                return 'for_sale'
            elif label.startswith('rec.autos'):
                return 'autos'
            elif label.startswith('rec.motorcycles'):
                return 'motorcycles'
            elif label.startswith('rec.sport.baseball'):
                return 'baseball'
            elif label.startswith('rec.sport.hockey'):
                return 'hockey'
            elif label.startswith('sci.crypt'):
                return 'cryptography'
            elif label.startswith('sci.electronics'):
                return 'electronics'
            elif label.startswith('sci.med'):
                return 'medicine'
            elif label.startswith('sci.space'):
                return 'space'
            elif label.startswith('soc.religion.christian'):
                return 'christianity'
            elif label.startswith('talk.politics.guns'):
                return 'politics_guns'
            elif label.startswith('talk.politics.mideast'):
                return 'politics_middle_east'
            elif label.startswith('talk.politics.misc'):
                return 'politics'
            elif label.startswith('talk.religion.misc'):
                return 'religion'
            else:
                return label

        # Apply the label mapping to rename the labels
        labels = [map_label(label) for label in labels]

        # Print the unique labels
        unique_labels = list(set(labels))
        print(unique_labels)

        # Count the occurrences of each renamed label
        label_counts = Counter(labels)
        atheism_count = label_counts['atheism']
        graphics_count = label_counts['graphics']
        windows_count = label_counts['windows']
        pc_hardware_count = label_counts['pc_hardware']
        mac_hardware_count = label_counts['mac_hardware']
        windows_x_count = label_counts['windows_x']
        for_sale_count = label_counts['for_sale']
        autos_count = label_counts['autos']
        motorcycles_count = label_counts['motorcycles']
        baseball_count = label_counts['baseball']
        hockey_count = label_counts['hockey']
        cryptography_count = label_counts['cryptography']
        electronics_count = label_counts['electronics']
        medicine_count = label_counts['medicine']
        space_count = label_counts['space']
        christianity_count = label_counts['christianity']
        politics_guns_count = label_counts['politics_guns']
        politics_middle_east_count = label_counts['politics_middle_east']
        politics_count = label_counts['politics']
        religion_count = label_counts['religion']
        print("Number of news about Atheism:", atheism_count)
        print("Number of news about Graphics:", graphics_count)
        print("Number of news about Windows:", windows_count)
        print("Number of news about PC_Hardware:", pc_hardware_count)
        print("Number of news about Mac_Hardware:", mac_hardware_count)
        print("Number of news about Windows_X:", windows_x_count)
        print("Number of news about For_Sale:", for_sale_count)
        print("Number of news about Autos:", autos_count)
        print("Number of news about Motorcycles:", motorcycles_count)
        print("Number of news about Baseball:", baseball_count)
        print("Number of news about Hockey:", hockey_count)
        print("Number of news about Cryptography:", cryptography_count)
        print("Number of news about Electronics:", electronics_count)
        print("Number of news about Medicine:", medicine_count)
        print("Number of news about Space:", space_count)
        print("Number of news about Christianity:", christianity_count)
        print("Number of news about Politics_Guns:", politics_guns_count)
        print("Number of news about Politics_Middle_East:", politics_middle_east_count)
        print("Number of news about Politics:", politics_count)
        print("Number of news about Religion:", religion_count)

        # Count the occurrences of each label type
        labels_counts = Counter(labels)

        #########################

        # Retrieve the values of the input pins
        num_classes = self.num_classes.getData()
        max_words = self.max_words.getData()
        max_length = self.max_length.getData()
        test_size = self.test_size.getData()

        # Tokenize the text
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)

        # Encode the text sequences
        sequences = tokenizer.texts_to_sequences(texts)

        # Pad sequences to a fixed length
        x_data = pad_sequences(sequences, maxlen=max_length)

        # Convert labels to one-hot encoding
        label_classes = list(set(labels))
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        print("labels:", labels)
        print("label_to_index:", label_to_index)

        y_data = [label_to_index[label] for label in labels]
        y_data = to_categorical(y_data, num_classes)

        # Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)

        # Set the data for the output pins
        self.dataset.setData((texts, labels))
        self.x_train.setData(x_train)
        self.y_train.setData(y_train)
        self.x_test.setData(x_test)
        self.y_test.setData(y_test)

        ############################## COUNT - count train

        # Convert the one-hot encoded labels back to their original label names
        y_train_labels = [label_classes[np.argmax(label)] for label in y_train]
        # Count the occurrences of each label within the y_train_labels list
        y_train_label_counts = Counter(y_train_labels)

        # Count the occurrences of the labels within the y_train_labels list
        atheism_count_train = y_train_label_counts['atheism']
        graphics_count_train = y_train_label_counts['graphics']
        windows_count_train = y_train_label_counts['windows']
        pc_hardware_count_train = y_train_label_counts['pc_hardware']
        mac_hardware_count_train = y_train_label_counts['mac_hardware']
        windows_x_count_train = y_train_label_counts['windows_x']
        for_sale_count_train = y_train_label_counts['for_sale']
        autos_count_train = y_train_label_counts['autos']
        motorcycles_count_train = y_train_label_counts['motorcycles']
        baseball_count_train = y_train_label_counts['baseball']
        hockey_count_train = y_train_label_counts['hockey']
        cryptography_count_train = y_train_label_counts['cryptography']
        electronics_count_train = y_train_label_counts['electronics']
        medicine_count_train = y_train_label_counts['medicine']
        space_count_train = y_train_label_counts['space']
        christianity_count_train = y_train_label_counts['christianity']
        politics_guns_count_train = y_train_label_counts['politics_guns']
        politics_middle_east_count_train = y_train_label_counts['politics_middle_east']
        politics_count_train = y_train_label_counts['politics']
        religion_count_train = y_train_label_counts['religion']

        ########### count test
        # Convert the one-hot encoded labels back to their original label names
        y_test_labels = [label_classes[np.argmax(label)] for label in y_test]
        # Count the occurrences of each label within the y_test_labels list
        y_test_label_counts = Counter(y_test_labels)

        # Count the occurrences of the labels within the y_test_labels list
        atheism_count_test = y_test_label_counts['atheism']
        graphics_count_test = y_test_label_counts['graphics']
        windows_count_test = y_test_label_counts['windows']
        pc_hardware_count_test = y_test_label_counts['pc_hardware']
        mac_hardware_count_test = y_test_label_counts['mac_hardware']
        windows_x_count_test = y_test_label_counts['windows_x']
        for_sale_count_test = y_test_label_counts['for_sale']
        autos_count_test = y_test_label_counts['autos']
        motorcycles_count_test = y_test_label_counts['motorcycles']
        baseball_count_test = y_test_label_counts['baseball']
        hockey_count_test = y_test_label_counts['hockey']
        cryptography_count_test = y_test_label_counts['cryptography']
        electronics_count_test = y_test_label_counts['electronics']
        medicine_count_test = y_test_label_counts['medicine']
        space_count_test = y_test_label_counts['space']
        christianity_count_test = y_test_label_counts['christianity']
        politics_guns_count_test = y_test_label_counts['politics_guns']
        politics_middle_east_count_test = y_test_label_counts['politics_middle_east']
        politics_count_test = y_test_label_counts['politics']
        religion_count_test = y_test_label_counts['religion']

        ##############################

        # Update the number of classes and test size in the custom layout
        num_classes = self.num_classes.getData()
        test_size = self.test_size.getData()
        nr_classes_testsize = f"Number of classes (default is 20): {num_classes}\nTest set size: {test_size * 100}%\nTrain set size: {100-test_size * 100}%"
        print(nr_classes_testsize)
        max_words2 = f"Max_words is: {max_words}"
        print(max_words2)
        max_length2 = f"Max_length is: {max_length}"
        print(max_length2)

        # Convert x_train and x_test to lists of strings
        x_train2 = x_train.tolist()
        x_test2 = x_test.tolist()

        # Compute the total number of tokens
        total_tokens_train = np.sum([len(sequence) for sequence in x_train2])
        total_tokens_test = np.sum([len(sequence) for sequence in x_test2])
        total_tokens_dataset = total_tokens_train + total_tokens_test

        total_tokens_train2 = f"Total_tokens in train set is: {total_tokens_train}"
        print(total_tokens_train2)
        total_tokens_test2 = f"Total_tokens in test set is: {total_tokens_test}"
        print(total_tokens_test2)
        total_tokens_dataset2 = f"Total_tokens in dataset is: {total_tokens_dataset}"
        print(total_tokens_dataset2)

        # Dataset description
        dataset_description = "The 20 Newsgroups dataset is a collection of approximately 18,000 newsgroup posts on 20 topics. These topics include various subjects such as computer hardware, software, politics, religion, sports, and more. The dataset is commonly used for text classification and natural language processing tasks. Each document is represented as a string, and the corresponding labels range from 0 to 19, representing the 20 different newsgroups."
        self.dataset_description.setData(dataset_description)
        # print(dataset_description)

        def convert_string_to_image(string):
            # Create a figure and plot the string as text
            plt.figure(figsize=(10, 4))
            plt.axis('off')
            plt.text(0.5, 0.5, string, fontsize=18, ha='center', va='center', wrap=True, fontname='DejaVu Sans')

            # Save the figure as a PNG file
            image_pathNEWS = 'Dataset_descriptionNEWS.png'
            plt.savefig(image_pathNEWS)
            plt.close()

            return image_pathNEWS

        image_pathNEWS = convert_string_to_image(dataset_description)
        self.dataset_description.setData(image_pathNEWS)

        # Dataset examples
        texts2 = texts[:2][:20]

        def convert_strings_to_image(strings):
            # Create a figure with subplots for each string
            fig, axs = plt.subplots(len(strings), 1, figsize=(16, 10))

            # Iterate over the strings and plot them as text in each subplot
            for i, string in enumerate(strings):
                axs[i].axis('off')
                axs[i].text(0.01, 0.01, string, fontsize=14, ha='left', va='center', wrap=True, fontname='DejaVu Sans')

            # Adjust the spacing between subplots
            plt.subplots_adjust(hspace=2)

            # Save the figure as a PNG file
            image_pathNEWS2 = 'Dataset_examplesNEWS.png'
            plt.savefig(image_pathNEWS2)
            plt.close()

            return image_pathNEWS2

        image_pathNEWS2 = convert_strings_to_image(texts2)
        self.examples.setData(image_pathNEWS2)

        # Dataset Dimension Table

        # Update the dataset information in the custom layout
        train_data_shape = x_train.shape if isinstance(x_train, np.ndarray) else (len(x_train),)
        test_data_shape = x_test.shape if isinstance(x_test, np.ndarray) else (len(x_test),)
        train_label_shape = y_train.shape if isinstance(y_train, np.ndarray) else (len(y_train),)
        test_label_shape = y_test.shape if isinstance(y_test, np.ndarray) else (len(y_test),)

        v1=train_data_shape
        v2=train_label_shape
        v3=test_data_shape
        v4=test_label_shape

        dimension = pd.DataFrame({
            'Train data \n (n_samples, n_features)': [v1],
             'Train labels \n (n_samples, n_labels)': [v2],
             'Test data \n (n_samples, n_features)': [v3],
             'Test labels \n (n_samples, n_labels)': [v4]
        })

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(18, 4))

        # Create the table and adjust font size and cell padding
        table15 = plt.table(cellText=dimension.values, colLabels=dimension.columns, cellLoc='center', loc='center')

        # Adjust font size
        table15.set_fontsize(18)

        # Color the cells in the first row with light blue
        for cell in table15.get_celld().values():
            if cell.get_text().get_text() in dimension.columns:
                cell.set_facecolor('lightblue')

        # Enable content wrapping in cells
        for cell in table15.get_celld().values():
            cell.set_text_props(wrap=True)

        # Adjust cell size
        table15.scale(1, 9.5)  # Increase the cell size by a factor of 1.5

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('dataset_dimensionNEWS.png')
        plt.close()

        self.dataset_dimension.setData('dataset_dimensionNEWS.png')

        # Parameters

        parameter = pd.DataFrame({
            'Nr. Classes': [num_classes],
            'Train set \nsize': [f'{100 - test_size * 100}%'],
            'Test set \nsize': [f'{test_size * 100}%'],
            'Max words': [max_words],
            'Max Length': [max_length]
        })

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(18, 4))

        # Create the table and adjust font size and cell padding
        table22 = plt.table(cellText=parameter.values, colLabels=parameter.columns, cellLoc='center', loc='center')

        # Adjust font size
        table22.set_fontsize(18)

        # Color the cells in the first row with light blue
        for cell in table22.get_celld().values():
            if cell.get_text().get_text() in parameter.columns:
                cell.set_facecolor('lightblue')

        # Enable content wrapping in cells
        for cell in table22.get_celld().values():
            cell.set_text_props(wrap=True)

        # Adjust cell size
        table22.scale(1, 6.5)  # Increase the cell size by a factor of 1.5

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('parametersNEWS.png')
        plt.close()

        self.parameters.setData('parametersNEWS.png')

        # Tokenization data

        tokenizat = pd.DataFrame({
            'Dataset Total \ntokens': [f'{total_tokens_dataset:,}'],
            'Train set \ntokens': [f'{total_tokens_train:,}'],
            'Test set \ntokens': [f'{total_tokens_test:,}']
        })

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(10, 8))

        # Create the table and adjust font size and cell padding
        table17 = plt.table(cellText=tokenizat.values, colLabels=tokenizat.columns, cellLoc='center', loc='center')

        # Adjust font size
        table17.set_fontsize(18)

        # Color the cells in the first row with light blue
        for cell in table17.get_celld().values():
            if cell.get_text().get_text() in tokenizat.columns:
                cell.set_facecolor('lightblue')

        # Enable content wrapping in cells
        for cell in table17.get_celld().values():
            cell.set_text_props(wrap=True)

        # Adjust cell size
        table17.scale(1, 6.5)  # Increase the cell size by a factor of 1.5

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('tokenizationNEWS.png')
        plt.close()

        self.tokenization.setData('tokenizationNEWS.png')


        # Counter

        # Create a DataFrame to display the counts
        counter2 = pd.DataFrame({
            'Tot. News': [label_counts[label] for label in unique_labels],
            'Tot. News in Train': [y_train_label_counts[label] for label in unique_labels],
            'Tot. News in Test': [y_test_label_counts[label] for label in unique_labels],
            'Classes': [[label] for label in unique_labels]
        })

        # Sort the DataFrame based on the 'Tot. News' column in descending order
        counter2 = counter2.sort_values('Tot. News', ascending=False)

        # Append the totals row to the DataFrame
        totals_row = {
            'Tot. News': f"Tot. {sum(label_counts.values())}",
            'Tot. News in Train': f"Tot. {sum(y_train_label_counts.values())}",
            'Tot. News in Test': f"Tot. {sum(y_test_label_counts.values())}",
            'Classes': f"Tot. {len(label_counts)} Classes"
        }

        # Create a DataFrame for the totals row and specify the index
        totals_df = pd.DataFrame(totals_row, index=[len(unique_labels)])

        # Concatenate the totals_df with counter2 DataFrame
        counter2 = pd.concat([counter2, totals_df], ignore_index=True)

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(14, 10))

        # Create the table and adjust font size and cell padding
        table18 = plt.table(cellText=counter2.values, colLabels=counter2.columns, cellLoc='center', loc='center')

        # Adjust font size
        table18.set_fontsize(12)

        # Color the cells in the first row with light blue
        for cell in table18.get_celld().values():
            if cell.get_text().get_text() in counter2.columns:
                cell.set_facecolor('lightblue')

        # Enable content wrapping in cells
        for cell in table18.get_celld().values():
            cell.set_text_props(wrap=True)

        # Adjust cell size
        table18.scale(1, 2.5)  # Increase the cell size by a factor of 1.5

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('countersNEWS.png')
        plt.close()

        self.counters.setData('countersNEWS.png')

        #####################################################

        # Check if values are set, if not prompt the user
        if not self.num_classes.hasConnections() or not self.max_words.hasConnections() or not self.max_length.hasConnections() or not self.test_size.hasConnections():
            self.promptVariables()

        num_classes = self.num_classes.getData()
        max_words = self.max_words.getData()
        max_length = self.max_length.getData()
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
            ("max_length", self.max_length),
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
