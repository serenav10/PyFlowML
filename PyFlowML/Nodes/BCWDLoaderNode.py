from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from tensorflow import keras
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets, QtCore
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from PySide2.QtWidgets import QLabel, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QPushButton
from PySide2.QtCore import Qt
from PyFlow.Core.Common import *
from sklearn import metrics
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from matplotlib.table import table
from collections import Counter
import pandas as pd

# Breast Cancer Wisconsin Diagnostic dataset
class BCWDLoaderNode(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False

        ####################### prompt only on refreshing this node
        async def refresh_node(node):
            # Check if the node is of type 'BCWDLoaderNode'
            if node.type == 'BCWDLoaderNode':
                # Show the prompt for 'BCWDLoaderNode'
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')

        #######################
        super(BCWDLoaderNode, self).__init__(name)

        # Define the output and input pins
        self.num_classes = self.createInputPin('num_classes', 'IntPin')
        self.test_size = self.createInputPin("test_size", 'FloatPin')

        self.x_train = self.createOutputPin("x_train", 'AnyPin')
        self.y_train = self.createOutputPin("y_train", 'AnyPin')
        self.x_test = self.createOutputPin("x_test", 'AnyPin')
        self.y_test = self.createOutputPin("y_test", 'AnyPin')
        self.dataset = self.createOutputPin("dataset", 'AnyPin')
        self.feature_names = self.createOutputPin("feature_names", 'StringPin')
        self.dataset_description = self.createOutputPin("dataset_description", 'StringPin')
        self.dataset_dimension = self.createOutputPin("dataset_dimension", 'StringPin')
        self.parameters = self.createOutputPin("parameters", 'StringPin')
        self.counters = self.createOutputPin("counters", 'StringPin')

        # Enable the allowAny option for the output pins
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.y_train.enableOptions(PinOptions.AllowAny)
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.y_test.enableOptions(PinOptions.AllowAny)
        self.dataset.enableOptions(PinOptions.AllowAny)

        # Set default values for prompt
        self.num_classes.setData(2)
        self.test_size.setData(0.2)

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('IntPin')
        helper.addInputDataType('IntPin')  # num_classes
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

            test_size_dialog = QtWidgets.QInputDialog()
            test_size_dialog.setInputMode(QtWidgets.QInputDialog.DoubleInput)
            test_size_dialog.setDoubleRange(0.0, 1.0)
            test_size_dialog.setWindowTitle(window_name)
            test_size_dialog.setLabelText("Enter value for 'test_size':")
            test_size_dialog.setOkButtonText("OK")
            test_size_dialog.setCancelButtonText("Cancel")

            # Set the width of the prompt windows
            num_classes_dialog.resize(window_width, num_classes_dialog.height())
            test_size_dialog.resize(window_width, test_size_dialog.height())

            num_classes_dialog.setIntValue(num_classes_default)
            if num_classes_dialog.exec_() == QtWidgets.QDialog.Accepted:
                num_classes_value = num_classes_dialog.intValue()
            else:
                # User canceled, so show a message and exit the prompt
                num_classes_dialog.reject()  # Close the dialog
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
            self.test_size.setData(test_size_value)

            # Refresh the data
            self.data_refreshed = True
            self.compute()
            return

        else:
            # User chose not to refresh the data
            self.num_classes.setData(2)
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
        # Load the Breast Cancer dataset
        dataset = load_breast_cancer()
        x_data = dataset.data
        y_data = dataset.target
        feature_names = dataset.feature_names

        # Create a mapping of unique labels
        classes_labels = {0: "Malignant", 1: "Benign"}

        # Map the labels to their corresponding names
        unique_labels0 = np.array([classes_labels.get(label, "Other") for label in y_data])
        unique_labels = list(set(unique_labels0))
        # Convert unique_labels to two separate strings
        label1 = str(unique_labels[0])
        label2 = str(unique_labels[1])

        # Get the test size from the input pin
        test_size = self.test_size.getData()

        # Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)

        # Set the data for the output pins
        self.x_train.setData(x_train)
        self.y_train.setData(y_train)
        self.x_test.setData(x_test)
        self.y_test.setData(y_test)
        self.feature_names.setData(feature_names)

        # Update the number of classes and test size in the custom layout
        num_classes = self.num_classes.getData()
        test_size = self.test_size.getData()
        nr_classes_testsize = f"Number of classes (default is 2): {num_classes}\nTest set size: {test_size * 100}%\nTrain set size: {100-test_size * 100}%"
        print(nr_classes_testsize)

        # Count the number of benign and malignant instances
        num_benign = np.count_nonzero(y_data == 1)
        num_malignant = np.count_nonzero(y_data == 0)
        count_full = f"Number of benign instances is {num_benign}\nNumber of malignant instances is {num_malignant}"
        print(count_full)

        # Count the number of benign and malignant instances in the train set
        num_benign_train = np.count_nonzero(y_train == 1)
        num_malignant_train = np.count_nonzero(y_train == 0)
        count_train = f"Number of benign instances in train set is {num_benign_train}\nNumber of malignant instances in train set is {num_malignant_train}"
        print(count_train)

        # Count the number of benign and malignant instances in the test set
        num_benign_test = np.count_nonzero(y_test == 1)
        num_malignant_test = np.count_nonzero(y_test == 0)
        count_test = f"Number of benign instances in test set is {num_benign_test}\nNumber of malignant instances in test set is {num_malignant_test}"
        print(count_test)

        # Set the data for the output pins
        self.dataset.setData((x_data, y_data))
        self.x_train.setData(x_train)
        self.y_train.setData(y_train)
        self.x_test.setData(x_test)
        self.y_test.setData(y_test)

        # Dataset description
        dataset_descriptionBCWD = "The Breast Cancer Wisconsin Diagnostic dataset (BCWD) was collected from fine needle aspirates FNA (Fine Needle Aspiration) of breast masses, and the goal was to create a diagnostic system to distinguish between benign and malignant breast tumors based on cytological features extracted from the digitized images of the FNA samples. The dataset consists of measurements computed from digitized images of FNA samples, including various characteristics of the cell nuclei present in the images. The dataset has been widely used as a benchmark in machine learning and pattern recognition research for breast cancer classification tasks. The BCWD contains the following 30 measurements/attributes (float type): 1) Radius Mean, 2) Texture Mean, 3) Perimeter Mean, 4) Area Mean, 5) Smoothness Mean, 6) Compactness Mean, 7) Concavity Mean, 8) Concave Points Mean, 9) Symmetry Mean, 10) Fractal Dimension Mean, 11) Radius SE (Standard Error), 12) Texture SE (Standard Error), 13) Perimeter SE (Standard Error), 14) Area SE (Standard Error), 15) Smoothness SE (Standard Error), 16) Compactness SE (Standard Error), 17) Concavity SE (Standard Error), 18) Concave Points SE (Standard Error), 19) Symmetry SE (Standard Error), 20) Fractal Dimension SE (Standard Error), 21) Radius Worst, 22) Texture Worst, 23) Perimeter Worst, 24) Area Worst, 25) Smoothness Worst, 26) Compactness Worst, 27) Concavity Worst, 28) Concave Points Worst, 29) Symmetry Worst, 30) Fractal Dimension Worst"

        def convert_string_to_image(string):
            # Create a figure and plot the string as text
            plt.figure(figsize=(18, 6))
            plt.axis('off')
            plt.text(0.5, 0.5, string, fontsize=18, ha='center', va='center', wrap=True, fontname='DejaVu Sans')

            # Save the figure as a PNG file
            image_pathBCWD = 'Dataset_descriptionBCWD.png'
            plt.savefig(image_pathBCWD)
            plt.close()

            return image_pathBCWD

        image_pathBCWD = convert_string_to_image(dataset_descriptionBCWD)
        self.dataset_description.setData(image_pathBCWD)

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

        plt.savefig('dataset_dimensionBCWD.png')
        plt.close()

        self.dataset_dimension.setData('dataset_dimensionBCWD.png')

        # Parameters

        parameter = pd.DataFrame({
            'Nr. Classes': [num_classes],
            'Train set \nsize': [f'{100 - test_size * 100}%'],
            'Test set \nsize': [f'{test_size * 100}%']
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

        plt.savefig('parametersBCWD.png')
        plt.close()

        self.parameters.setData('parametersBCWD.png')

        # Counter

        counter = pd.DataFrame({
            'Tot. Benign': [num_benign],
            'Tot. Malignant': [num_malignant],
            'Tot. Benign\nin train': [num_benign_train],
            'Tot. Malignant\nin train': [num_malignant_train],
            'Tot. Benign\nin test': [num_benign_test],
            'Tot. Malignant\nin test': [num_malignant_test],
            'Classes': f"{label1}, \n {label2}"
        })

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(20, 4))

        # Create the table and adjust font size and cell padding
        table13 = plt.table(cellText=counter.values, colLabels=counter.columns, cellLoc='center', loc='center')

        # Adjust font size
        table13.set_fontsize(18)

        # Color the cells in the first row with light blue
        for cell in table13.get_celld().values():
            if cell.get_text().get_text() in counter.columns:
                cell.set_facecolor('lightblue')

        # Enable content wrapping in cells
        for cell in table13.get_celld().values():
            cell.set_text_props(wrap=True)

        # Adjust cell size
        table13.scale(1, 6.5)  # Increase the cell size by a factor of 1.5

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('countersBCWD.png')
        plt.close()

        self.counters.setData('countersBCWD.png')

        # Dataset feature_names

        def convert_strings_to_image(strings):
            # Create a figure with subplots for each string
            fig, axs = plt.subplots(len(strings), 1, figsize=(10, 12))

            # Iterate over the strings and plot them as text in each subplot
            for i, string in enumerate(strings):
                axs[i].axis('off')
                axs[i].text(0.01, 0.01, string, fontsize=18, ha='left', va='center', wrap=True, fontname='DejaVu Sans')

            # Adjust the spacing between subplots
            plt.subplots_adjust(hspace=1.5)

            # Save the figure as a PNG file
            image_pathBCWD = 'Dataset_examplesBCWD.png'
            plt.savefig(image_pathBCWD)
            plt.close()

            return image_pathBCWD

        image_pathBCWD = convert_strings_to_image(feature_names)
        self.feature_names.setData(image_pathBCWD)

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

        #####################################################

        # Check if values are set, if not prompt the user
        if not self.num_classes.hasConnections() or not self.max_words.hasConnections() or not self.max_length.hasConnections() or not self.test_size.hasConnections():
            self.promptVariables()

        num_classes = self.num_classes.getData()
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
            ("parameters", self.parameters),
            ("num_classes", self.num_classes),
            ("test_size", self.test_size),
            ("dataset", self.dataset),
            #("counters", self.counters),
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
