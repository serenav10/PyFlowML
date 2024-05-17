from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from tensorflow import keras
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
from PySide2.QtWidgets import QLabel, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QPushButton, QMessageBox
from PySide2.QtCore import Qt
from PyFlow.Core.Common import *
from sklearn import metrics
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from matplotlib.table import table
from collections import Counter
import os

class TextLoaderNode(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False
        self.parametersSet = False  # Initialize the attribute here
        self.messagesShown = False

####################### prompt only on refreshing this node
        async def refresh_node(node):
            if node.type == 'TextLoaderNode':
                # Show the prompt
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')
#######################

        super(TextLoaderNode, self).__init__(name)

        # Define the input and output pins
        self.num_classes = self.createInputPin("Number of Classes", 'IntPin')
        self.test_size = self.createInputPin("Dimension of the Test set", 'FloatPin')
        self.max_words = self.createInputPin("Maximum Number of Words", 'IntPin')
        self.max_length = self.createInputPin("Maximum Number of Characters", 'IntPin')

        self.x_train = self.createOutputPin("Data for Training", 'AnyPin')
        self.y_train = self.createOutputPin("Labels for Training", 'AnyPin')
        self.x_test = self.createOutputPin("Data for Test", 'AnyPin')
        self.y_test = self.createOutputPin("Labels for Test", 'AnyPin')
        self.dataset = self.createOutputPin("Original Dataset", 'AnyPin')
        self.classes_name = self.createOutputPin("Name of Classes", 'AnyPin')

        # Enable the allowAny option for the output pins
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.y_train.enableOptions(PinOptions.AllowAny)
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.y_test.enableOptions(PinOptions.AllowAny)
        self.dataset.enableOptions(PinOptions.AllowAny)
        self.classes_name.enableOptions(PinOptions.AllowAny)

        self.dataset_dimension = self.createOutputPin("Dimension of Dataset", 'StringPin')
        self.tokenization = self.createOutputPin("Tokenization", 'StringPin')
        self.parameters = self.createOutputPin("Parameters", 'StringPin')
        self.examples = self.createOutputPin("Examples in the collection", 'StringPin')
        self.counters = self.createOutputPin("Distribution of Dataset", 'StringPin')

        # Set default values for prompt
        self.num_classes.setData(2)
        self.max_words.setData(100)
        self.max_length.setData(50)
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
        self.parametersSet = True

        if not self.messagesShown:
            # Information dialog
            info_dialog = QMessageBox()
            info_dialog.setWindowTitle("File Upload Instructions")
            info_dialog.setText(
                "Please, upload a txt or csv file containing columns with headers, separated by a tabular (\\t). The first column contains the target.")
            info_dialog.setStandardButtons(QMessageBox.Ok)
            info_dialog.exec_()

            # Settings dialog
            refresh_dialog = QtWidgets.QMessageBox()
            refresh_dialog.setWindowTitle("Settings")
            refresh_dialog.setText("Do you want to set parameters? If already done, skip this and choose No")
            refresh_dialog.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            refresh_dialog.setDefaultButton(QtWidgets.QMessageBox.No)

            response = refresh_dialog.exec_()

            # Set the flag to True after the message boxes have been shown
            self.messagesShown = True

            if response == QtWidgets.QMessageBox.Yes:

                # Prompt the user to set the values
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
                num_classes_dialog.setLabelText("Enter value for the Number of Classes:")
                num_classes_dialog.setOkButtonText("OK")
                num_classes_dialog.setCancelButtonText("Cancel")

                max_words_dialog = QtWidgets.QInputDialog()
                max_words_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)
                max_words_dialog.setIntRange(-2147483648, 2147483647)
                max_words_dialog.setWindowTitle(window_name)
                max_words_dialog.setLabelText("Enter value for Maximum Number of Words:")
                max_words_dialog.setOkButtonText("OK")
                max_words_dialog.setCancelButtonText("Cancel")

                max_length_dialog = QtWidgets.QInputDialog()
                max_length_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)
                max_length_dialog.setIntRange(-2147483648, 2147483647)
                max_length_dialog.setWindowTitle(window_name)
                max_length_dialog.setLabelText("Enter value for Maximum Number of Characters:")
                max_length_dialog.setOkButtonText("OK")
                max_length_dialog.setCancelButtonText("Cancel")

                test_size_dialog = QtWidgets.QInputDialog()
                test_size_dialog.setInputMode(QtWidgets.QInputDialog.DoubleInput)
                test_size_dialog.setDoubleRange(0.0, 1.0)
                test_size_dialog.setWindowTitle(window_name)
                test_size_dialog.setLabelText("Enter value for the Dimension of the Test set:")
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
                    self.showDataNotRefreshedMessage()
                    return

                max_words_dialog.setIntValue(max_words_default)
                if max_words_dialog.exec_() == QtWidgets.QDialog.Accepted:
                    max_words_value = max_words_dialog.intValue()
                else:
                    # User canceled, so show a message and exit the prompt
                    self.showDataNotRefreshedMessage()
                    return

                max_length_dialog.setIntValue(max_length_default)
                if max_length_dialog.exec_() == QtWidgets.QDialog.Accepted:
                    max_length_value = max_length_dialog.intValue()
                else:
                    # User canceled, so show a message and exit the prompt
                    self.showDataNotRefreshedMessage()
                    return

                test_size_dialog.setDoubleValue(test_size_default)
                if test_size_dialog.exec_() == QtWidgets.QDialog.Accepted:
                    test_size_str = str(test_size_dialog.doubleValue())
                    test_size_value = float(test_size_str)
                else:
                    # User canceled, so show a message and exit the prompt
                    self.showDataNotRefreshedMessage()
                    return

                # Set the data of pins with the entered values
                self.num_classes.setData(num_classes_value)
                self.max_words.setData(max_words_value)
                self.max_length.setData(max_length_value)
                self.test_size.setData(test_size_value)
                self.showExecutionCompletedMessage()
                return

            else:
                # User chose not to refresh the data
                return

        else:
            # User chose not to refresh the data
            self.num_classes.setData(2)
            self.test_size.setData(0.2)
            self.max_words.setData(100)
            self.max_length.setData(50)
            return


    def showDataNotRefreshedMessage(self):
        message_box = QtWidgets.QMessageBox()
        message_box.setWindowTitle("Settings")
        message_box.setText("Default parameters have been set.")
        message_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        message_box.exec_()

    def showExecutionCompletedMessage(self):
        message_box = QtWidgets.QMessageBox()
        message_box.setWindowTitle("Information")
        message_box.setText("Executing... Please wait for the node's green outline to appear")
        message_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        message_box.exec_()

    def compute(self, *args, **kwargs):
        # Prompt the user to set the parameters first
        self.promptVariables()

        # Check if the parameters have been set
        if not self.parametersSet:
            # Handle the case where parameters are not yet set
            return

        # File paths
        txt_file_path = r"C:\Users\sere\PyFlowOpenCv\PyFlow\Packages\PyFlowML\Datasets\data.txt"
        csv_file_path = r"C:\Users\sere\PyFlowOpenCv\PyFlow\Packages\PyFlowML\Datasets\data.csv"

        # Check if data.txt exists
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        elif os.path.exists(csv_file_path):
            # If data.csv exists, handle it accordingly
            # For example, read it with pandas or line by line
            # This depends on how you want to process the CSV data
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()  # or use pandas or other method
        else:
            raise FileNotFoundError("Neither data.txt nor data.csv could be found.")

        # Process the lines to extract text and labels
        texts = []
        labels = []
        for i, line in enumerate(lines):
            if i == 0:  # Skip the first row
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                labels.append(parts[0].strip())
                texts.append(parts[1].strip())

        # Get the classes names
        classes_name = list(set(labels))  # Unique class names
        classes_list = list(enumerate(classes_name))  # Indexed classes

        ######################### COUNT

        # Unique labels and their counts
        unique_labels = list(set(labels))
        label_counts = Counter(labels)

        # Print the first five elements of texts
        if len(texts) > 0:
            print("First five texts:", texts[:5])

        # Print the unique labels
        print("Unique labels:", unique_labels)

        # Dynamically print counts for each label
        for label in unique_labels:
            print(f"Number of '{label}' classes:", label_counts[label])

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
        #print(sequences)

        # Pad sequences to a fixed length
        x_train = pad_sequences(sequences, maxlen=max_length)
        x_test = pad_sequences(sequences, maxlen=max_length)

        # Pad sequences to a fixed length (for dataset output)
        x_data = pad_sequences(sequences, maxlen=max_length)
        #print(x_data)

        # Convert labels to one-hot encoding
        label_classes = list(set(labels))
        label_to_index = {label: index for index, label in enumerate(label_classes)}
        y_train = [label_to_index[label] for label in labels]
        y_train = to_categorical(y_train, num_classes)

        # Convert labels to one-hot encoding (for dataset output)
        y_data = [label_to_index[label] for label in labels]
        y_data = to_categorical(y_data, num_classes)

        # Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=42)

        # Set the data for the output pins
        #self.dataset.setData((x_data, y_data))
        self.dataset.setData((texts, labels))
        self.x_train.setData(x_train)
        self.y_train.setData(y_train)
        self.x_test.setData(x_test)
        self.y_test.setData(y_test)
        self.classes_name.setData(classes_name)


        ############################## COUNT
        # Count the occurrences of each label type in y_train
        y_train_labels = [label_classes[np.argmax(label)] for label in y_train]
        y_train_label_counts = Counter(y_train_labels)

        # Dynamically print counts for each label in y_train
        for label in label_classes:
            print(f"Number of {label} classes in y_train:", y_train_label_counts[label])

        # Count the occurrences of each label type in y_test
        y_test_labels = [label_classes[np.argmax(label)] for label in y_test]
        y_test_label_counts = Counter(y_test_labels)

        # Dynamically print counts for each label in y_test
        for label in label_classes:
            print(f"Number of {label} classes in y_test:", y_test_label_counts[label])

        ##############################

        # Update the number of classes, test size, max words and lengths in the custom layout
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
        v1=x_train.shape
        v2=y_train.shape
        v3=x_test.shape
        v4=y_test.shape

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
            'Max words': [max_words],
            'Max Length': [max_length]
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

        # Calculate total counts for dataset, train set, and test set
        total_dataset_count = sum(label_counts.values())
        total_train_count = sum(y_train_label_counts.values())
        total_test_count = sum(y_test_label_counts.values())

        # Create a dynamic DataFrame for counters
        counter_data = []
        for label in label_classes:
            dataset_percentage = round((label_counts[label] / total_dataset_count) * 100, 2)
            train_percentage = round((y_train_label_counts[label] / total_train_count) * 100, 2)
            test_percentage = round((y_test_label_counts[label] / total_test_count) * 100, 2)

            counter_data.append({
                'Target variable': label,
                'Total in Dataset': label_counts[label],
                'Total in Train Set': y_train_label_counts[label],
                'Total in Test Set': y_test_label_counts[label],
                'Dataset %': f"{dataset_percentage}%",
                'Train Set %': f"{train_percentage}%",
                'Test Set %': f"{test_percentage}%"
            })

        counter_df = pd.DataFrame(counter_data)

        # Generate table
        fig_width = max(10, len(counter_df.columns) * 2)  # Adjust the width based on number of columns
        fig_height = max(6, len(counter_df) * 0.5)  # Adjust the height based on number of rows

        plt.figure(figsize=(fig_width, fig_height))
        ax = plt.gca()
        ax.axis('off')
        table = ax.table(cellText=counter_df.values, colLabels=counter_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        # Adjust cell color based on header row
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor('lightblue')
            else:
                cell.set_facecolor('white')

        table.scale(1, 2)  # Adjust cell size

        plt.savefig('counters.png')
        plt.close()

        self.counters.setData('counters.png')

        #####################################################

        # Check if values are set, if not prompt the user
        if not self.num_classes.hasConnections() or not self.max_words.hasConnections() or not self.max_length.hasConnections() or not self.test_size.hasConnections():
            self.promptVariables()

        num_classes = self.num_classes.getData()
        max_words = self.max_words.getData()
        max_length = self.max_length.getData()
        test_size = self.test_size.getData()

        # Show a message that the node has finished running
        finish_dialog = QMessageBox()
        finish_dialog.setWindowTitle("Information")
        finish_dialog.setText("Node successfully executed.")
        finish_dialog.setStandardButtons(QMessageBox.Ok)
        finish_dialog.exec_()

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
            ("tokenization", self.tokenization),
            ("parameters", self.parameters),
            ("num_classes", self.num_classes),
            ("max_words", self.max_words),
            ("max_length", self.max_length),
            ("test_size", self.test_size),
            ("examples", self.examples),
            ("dataset", self.dataset),
            ("classes_name", self.classes_name),
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
