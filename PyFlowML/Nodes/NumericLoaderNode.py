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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
import os
from sklearn.impute import SimpleImputer

class NumericLoaderNode(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False
        self.parametersSet = False  # Initialize the attribute here
        self.messagesShown = False

####################### prompt only on refreshing this node
        async def refresh_node(node):
            if node.type == 'NumericLoaderNode':
                # Show the prompt
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')
#######################

        super(NumericLoaderNode, self).__init__(name)

        # Define the input and output pins
        self.num_classes = self.createInputPin("Number of Classes", 'IntPin')
        self.test_size = self.createInputPin("Dimension of the Test set", 'FloatPin')

        self.x_train = self.createOutputPin("Data for Training", 'AnyPin')
        self.y_train = self.createOutputPin("Labels for Training", 'AnyPin')
        self.x_test = self.createOutputPin("Data for Test", 'AnyPin')
        self.y_test = self.createOutputPin("Labels for Test", 'AnyPin')
        self.dataset = self.createOutputPin("Original Dataset", 'AnyPin')
        self.classes_name = self.createOutputPin("Name of Classes", 'AnyPin')
        self.features_name = self.createOutputPin("Name of Features", 'AnyPin')
        self.features_list = self.createOutputPin("List of Features", 'StringPin')
        self.dataset_dimension = self.createOutputPin("Dimension of Dataset", 'StringPin')
        self.counters = self.createOutputPin("Distribution of Dataset", 'StringPin')
        self.parameters = self.createOutputPin("Parameters", 'StringPin')

        # Enable the allowAny option for the output pins
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.y_train.enableOptions(PinOptions.AllowAny)
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.y_test.enableOptions(PinOptions.AllowAny)
        self.dataset.enableOptions(PinOptions.AllowAny)
        self.classes_name.enableOptions(PinOptions.AllowAny)
        self.features_name.enableOptions(PinOptions.AllowAny)

        # Set default values for prompt
        self.num_classes.setData(0)
        self.test_size.setData(0.1)

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('IntPin')
        helper.addInputDataType('IntPin')  # num_classes
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
                test_size_default = self.test_size.getData()

                # Customize the name and size of the prompt window
                window_name = "Set parameters"
                window_width = 600

                num_classes_dialog = QtWidgets.QInputDialog()
                num_classes_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)
                num_classes_dialog.setIntRange(0, 100)
                num_classes_dialog.setWindowTitle(window_name)
                num_classes_dialog.setLabelText("Enter value for the Number of Classes:")
                num_classes_dialog.setOkButtonText("OK")
                num_classes_dialog.setCancelButtonText("Cancel")

                test_size_dialog = QtWidgets.QInputDialog()
                test_size_dialog.setInputMode(QtWidgets.QInputDialog.DoubleInput)
                test_size_dialog.setDoubleRange(0.00, 1.00)
                test_size_dialog.setWindowTitle(window_name)
                test_size_dialog.setLabelText("Enter value for the Dimension of the Test set:")
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
            #self.showDataNotRefreshedMessage()
            return

    def showDataNotRefreshedMessage(self):
        message_box = QtWidgets.QMessageBox()
        message_box.setWindowTitle("Settings")
        message_box.setText("Default parameters have been set. Please, consider that they might not fit your dataset.")
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

        # Check if the parameters have been set (especially num_classes)
        if not self.parametersSet:
            # Handle the case where parameters are not yet set
            return

        txt_file_path = r"C:\Users\sere\PyFlowOpenCv\PyFlow\Packages\PyFlowML\Datasets\data.txt"
        csv_file_path = r"C:\Users\sere\PyFlowOpenCv\PyFlow\Packages\PyFlowML\Datasets\data.csv"

        # Check if data.txt exists, otherwise load data.csv
        if os.path.exists(txt_file_path):
            df = pd.read_csv(txt_file_path, sep='\t')
        elif os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path, sep='\t')
        else:
            raise FileNotFoundError("Neither data.txt nor data.csv could be found.")

        #df.dropna(inplace=True)

        # Separate the data into numerical and categorical
        numerical_data = df.select_dtypes(include=[np.number])
        categorical_data = df.select_dtypes(exclude=[np.number])

        # Create imputers
        numerical_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        # Impute missing values
        numerical_data_imputed = numerical_imputer.fit_transform(numerical_data)
        categorical_data_imputed = categorical_imputer.fit_transform(categorical_data)

        # Convert imputed data back to a DataFrame
        numerical_data_imputed = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
        categorical_data_imputed = pd.DataFrame(categorical_data_imputed, columns=categorical_data.columns)

        # Combine the imputed data back into a single DataFrame
        df_imputed = pd.concat([numerical_data_imputed, categorical_data_imputed], axis=1)

        # Ensure the original order of columns is preserved
        df_imputed = df_imputed[df.columns]

        df = df_imputed

        feature_names = df.columns.tolist()
        features_list = df.columns.tolist()

        # Extract labels (first column) and features (remaining columns)
        labels = df.iloc[:, 0].values
        features_df = df.iloc[:, 1:]

        # Get unique classes and number of classes
        classes_name = np.unique(labels).tolist()
        num_classes = len(classes_name)

        # Process each feature based on its type
        processed_features = []
        new_feature_names = []

        for col in features_df.columns:
            if features_df[col].dtype == 'object' or features_df[col].dtype == 'category':
                # One-hot encoding for categorical columns
                encoder = OneHotEncoder(sparse=False)
                encoded = encoder.fit_transform(features_df[[col]])

                # Update feature names for one-hot encoded columns
                new_feature_names.extend([f"{col}_{category}" for category in encoder.categories_[0]])
                processed_features.append(encoded)
            else:
                # Scale numeric columns
                scaler = StandardScaler()
                scaled = scaler.fit_transform(features_df[[col]])
                new_feature_names.append(col)
                processed_features.append(scaled)

        # Concatenate processed features
        X_scaled = np.hstack(processed_features)

        # Split the data into train and test sets
        test_size = self.test_size.getData()  # Ensure this attribute exists in your class
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=test_size, random_state=42, stratify=labels)


        # Set the data for the output pins
        self.dataset.setData((X_scaled, labels))
        self.x_train.setData(x_train)
        self.y_train.setData(y_train)
        self.x_test.setData(x_test)
        self.y_test.setData(y_test)
        self.features_name.setData(new_feature_names)
        self.features_list.setData(features_df.columns.tolist())
        self.classes_name.setData(classes_name)

        ############################## COUNT
        # Count the occurrences of each label type in y_train
        y_train_label_counts = Counter(y_train)

        # Dynamically print counts for each label in y_train
        for label in classes_name:
            print(f"Number of {label} classes in y_train:", y_train_label_counts.get(label, 0))

        # Count the occurrences of each label type in y_test
        y_test_label_counts = Counter(y_test)

        # Dynamically print counts for each label in y_test
        for label in classes_name:
            print(f"Number of {label} classes in y_test:", y_test_label_counts.get(label, 0))
        ##############################

        # Update the number of classes, test size
        num_classes = self.num_classes.getData()
        test_size = self.test_size.getData()
        nr_classes_testsize = f"Number of classes (default is 2): {num_classes}\nTest set size: {test_size * 100}%\nTrain set size: {100-test_size * 100}%"
        print(nr_classes_testsize)

        # Dataset features_list

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
            image_path = 'Dataset_features_list.png'
            plt.savefig(image_path)
            plt.close()

            return image_path

        image_path = convert_strings_to_image(features_list)
        self.features_list.setData(image_path)

        # Your existing dataset dimension data
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

        # Create figure and axis for the table
        plt.figure(figsize=(18, 6))  # Adjust size as needed

        # Create the table
        table5 = plt.table(cellText=dimension.values, colLabels=dimension.columns, cellLoc='center', loc='center',
                           edges='closed')

        # Adjust the cell padding
        table5.auto_set_column_width(col=list(range(len(dimension.columns))))  # Adjust column width automatically

        # Adjust font size and cell scaling
        table5.auto_set_font_size(False)
        table5.set_fontsize(12)
        table5.scale(1, 2.5)  # Increase the vertical scale to provide more room for content

        # Style header cells
        header_colors = "#40466e"  # Dark blue color
        for (i, col) in enumerate(dimension.columns):
            table5[(0, i)].set_facecolor(header_colors)
            table5[(0, i)].set_text_props(weight='bold', color='white')
            table5[(0, i)].PAD = 0.05  # Increase padding for header cells

        # Style data cells
        data_cell_color = "#f1f1f2"  # Light gray color
        for i in range(1, len(dimension) + 1):
            for j in range(len(dimension.columns)):
                table5[(i, j)].set_facecolor(data_cell_color)

        # Add note below the table
        plt.figtext(0.5, 0.08,
                    "(Note: When categorical features are one-hot encoded, each unique category becomes a new feature.)",
                    wrap=True, horizontalalignment='center', fontsize=14)

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        # Save and close
        plt.savefig('dataset_dimension.png')
        plt.close()

        # Assuming you have a method to set data as in your original code
        self.dataset_dimension.setData('dataset_dimension.png')

        # Parameters

        parameter = pd.DataFrame({
            'Nr. Classes': [num_classes],
            'Train set \nsize': [f'{100 - test_size * 100}%'],
            'Test set \nsize': [f'{test_size * 100}%']
        })

        # Create figure and axis for the table
        plt.figure(figsize=(10, 4))

        # Create the table
        table6 = plt.table(cellText=parameter.values, colLabels=parameter.columns, cellLoc='center', loc='center',
                           edges='closed')

        # Adjust the cell padding
        table6.auto_set_column_width(col=list(range(len(parameter.columns))))  # Adjust column width automatically

        # Font size and cell scaling
        table6.auto_set_font_size(False)
        table6.set_fontsize(12)
        table6.scale(1, 2.5)  # Increase the vertical scale to provide more room for content

        # Style header cells
        header_color = "#40466e"  # Dark blue color
        for (i, col) in enumerate(parameter.columns):
            table6[(0, i)].set_facecolor(header_color)
            table6[(0, i)].set_text_props(weight='bold', color='white')
            table6[(0, i)].PAD = 0.05  # Increase padding for header cells

        # Style data cells
        data_cell_color = "#f1f1f2"  # Light gray color
        for i in range(1, len(parameter) + 1):
            for j in range(len(parameter.columns)):
                table6[(i, j)].set_facecolor(data_cell_color)

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        # Save and close
        plt.savefig('parameters.png')
        plt.close()

        # Assuming you have a method to set data as in your original code
        self.parameters.setData('parameters.png')

        # Calculate total counts for dataset, train set, and test set
        # Combine the counts from both y_train and y_test
        combined_label_counts = y_train_label_counts + y_test_label_counts
        total_dataset_count = sum(combined_label_counts.values())

        total_train_count = sum(y_train_label_counts.values())
        total_test_count = sum(y_test_label_counts.values())

        # Create a dynamic DataFrame for counters
        counter_data = []
        for label in classes_name:  # Assuming classes_name contains all unique labels
            dataset_percentage = round((combined_label_counts[label] / total_dataset_count) * 100, 2)
            train_percentage = round((y_train_label_counts[label] / total_train_count) * 100, 2)
            test_percentage = round((y_test_label_counts[label] / total_test_count) * 100, 2)

            counter_data.append({
                'Target variable': label,
                'Total in Dataset': combined_label_counts[label],
                'Total in Train Set': y_train_label_counts[label],
                'Total in Test Set': y_test_label_counts[label],
                'Dataset %': f"{dataset_percentage}%",
                'Train Set %': f"{train_percentage}%",
                'Test Set %': f"{test_percentage}%"
            })

        counter_df = pd.DataFrame(counter_data)

        # Generate table with adjusted sizing
        fig_width = max(10, len(counter_df.columns) * 2)  # Adjust the width based on number of columns
        fig_height = max(6, len(counter_df) * 0.5)  # Adjust the height based on number of rows

        plt.figure(figsize=(fig_width, fig_height))
        ax = plt.gca()
        ax.axis('off')
        table = ax.table(cellText=counter_df.values, colLabels=counter_df.columns, loc='center', cellLoc='center',
                         edges='closed')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Adjust cell size

        # Style header cells
        header_color = "#40466e"  # Dark blue color
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor(header_color)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor("#f1f1f2")  # Light gray color for data cells

        # Adjust the cell padding
        table.auto_set_column_width(col=list(range(len(counter_df.columns))))  # Adjust column width automatically

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        # Save and close
        plt.savefig('counters.png')
        plt.close()

        # Assuming you have a method to set data as in your original code
        self.counters.setData('counters.png')

        #####################################################

        # Check if values are set, if not prompt the user
        if not self.num_classes.hasConnections() or not self.test_size.hasConnections():
            self.promptVariables()

        num_classes = self.num_classes.getData()
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
            ("Data for Training", self.x_train),
            ("Labels for Training", self.y_train),
            ("Data for Test", self.x_test),
            ("Labels for Test", self.y_test),
            ("Dimension of dataset", self.dataset_dimension),
            ("Parameters", self.parameters),
            ("Number of Classes ", self.num_classes),
            ("Dimension of the Test set", self.test_size),
            ("Original Dataset", self.dataset),
            ("Name of Classes", self.classes_name),
            ("Distribution of Dataset", self.counters),
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
