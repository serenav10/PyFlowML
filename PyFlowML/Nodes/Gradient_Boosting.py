from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from PyFlow.Core.Common import *
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Packages.PyFlowBase.Pins.BoolPin import BoolPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from PySide2.QtWidgets import QLabel, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QPushButton, QMessageBox
from PySide2.QtCore import Qt
from matplotlib.table import table
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import itertools

class Gradient_Boosting(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False
        # self.parametersSet = False
        self.messagesShown = False

        ####################### prompt only on refreshing this node
        async def refresh_node(node):
            if node.type == 'Gradient_Boosting':
                # Show the prompt
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')

        #######################
        super(Gradient_Boosting, self).__init__(name)
        # Define the input pins
        self.x_train = self.createInputPin("Data for Training", 'AnyPin')
        self.y_train = self.createInputPin("Labels for Training", 'AnyPin')
        self.x_test = self.createInputPin("Data for Test", 'AnyPin')
        self.y_test = self.createInputPin("Labels for Test", 'AnyPin')
        self.classes_name = self.createInputPin('Name of Classes', 'AnyPin')

        # Define the output pin
        self.accuracy = self.createOutputPin("Accuracy", 'FloatPin')
        self.f1 = self.createOutputPin("F1 score", 'FloatPin')
        self.confusion_table = self.createOutputPin('Confusion Matrix Table', 'StringPin')
        self.plot = self.createOutputPin('Confusion Matrix Plot', 'StringPin')
        self.metrics_table = self.createOutputPin('Performance Metrics', 'StringPin')
        self.trained_model = self.createOutputPin("Trained Model", 'AnyPin')
        self.x_train_ = self.createOutputPin("Data for Training_", 'AnyPin')
        self.x_test_ = self.createOutputPin("Data for Test_", 'AnyPin')

        # Enable the allowAny option for the input and output pins
        self.y_test.enableOptions(PinOptions.AllowAny)
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.y_train.enableOptions(PinOptions.AllowAny)
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.trained_model.enableOptions(PinOptions.AllowAny)
        self.x_test_.enableOptions(PinOptions.AllowAny)
        self.x_train_.enableOptions(PinOptions.AllowAny)
        self.classes_name.enableOptions(PinOptions.AllowAny)


    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('AnyPin')
        helper.addOutputDataType('FloatPin')
        helper.addInputDataType('IntPin')
        helper.addOutputDataType('AnyPin')
        helper.addOutputDataType('StringPin')
        return helper

    def promptVariables(self):

        if not self.messagesShown:
            # Information dialog
            info_dialog = QMessageBox()
            info_dialog.setWindowTitle("Information")
            info_dialog.setText("Executing... Please wait for the node's green outline to appear.")
            info_dialog.setStandardButtons(QMessageBox.Ok)
            info_dialog.exec_()
            # Set the flag to True after showing the message
            self.messagesShown = True
        else:
            return

    def compute(self, *args, **kwargs):
        self.promptVariables()

        x_train = self.x_train.getData()
        y_train = self.y_train.getData()
        x_test = self.x_test.getData()
        y_test = self.y_test.getData()
        class_names = self.classes_name.getData()

        # Check for NaN values in x_train
        if np.isnan(x_train).any():
            # Handle NaN values, e.g., by filling with the mean
            x_train = np.nan_to_num(x_train)  # Replace NaN with 0 or use another method

        # Ensure x_train is a 2D array
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)

        # Scale x_train and x_test to ensure all features are non-negative
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Check if y_test is one-dimensional
        if len(y_test.shape) == 1:
            y_test_single_label = y_test
        else:
            y_test_single_label = np.argmax(y_test, axis=1)  # Convert multilabel indicator to single-label format

        # Convert y_train to a NumPy array
        y_train = np.array(y_train)

        # Reshape y_train to be a 1D array
        # y_train = np.argmax(y_train, axis=1)
        # If y_train is one-hot encoded, convert it back to the original format
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)

        # Create and train the Random Forest classifier
        clf = GradientBoostingClassifier()
        clf.fit(x_train, y_train)
        self.trained_model.setData(clf)
        self.x_train_.setData(x_train)
        self.x_test_.setData(x_test)

        # Start the timer
        start_time = time.time()

        # Predict the test set
        y_pred = clf.predict(x_test)

        # Check if y_pred is one-dimensional
        if len(y_pred.shape) == 1:
            y_pred_single_label = y_pred
        else:
            y_pred_single_label = np.argmax(y_pred, axis=1)

        # Convert y_test to a NumPy array and reshape if necessary
        y_test = np.array(y_test)

        # Check if y_pred is one-dimensional
        if len(y_pred.shape) == 1:
            y_pred_single_label = y_pred
        else:
            y_pred_single_label = np.argmax(y_pred, axis=1)

        # Convert y_test to a NumPy array and reshape if necessary
        y_test = np.array(y_test)
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate precision, recall, and f1 score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Set the accuracy, and f1 score as outputs
        self.accuracy.setData(accuracy)
        self.f1.setData(f1)

        # Convert y_test and y_pred to a NumPy array
        y_test_single_label = np.array(y_test)
        y_pred_single_label = np.array(y_pred)

        # Generate the confusion matrix
        conf = confusion_matrix(y_test_single_label, y_pred_single_label)

        # Calculate metrics for each class
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            true_positives = conf[i, i]
            false_negatives = np.sum(conf[i, :]) - true_positives
            false_positives = np.sum(conf[:, i]) - true_positives
            true_negatives = np.sum(conf) - true_positives - false_negatives - false_positives

            class_metrics[class_name] = {
                "True Positives": true_positives,
                "False Negatives": false_negatives,
                "False Positives": false_positives,
                "True Negatives": true_negatives
            }

        ################ CONFUSION MATRIX

        # Generate the confusion matrix
        conf = metrics.confusion_matrix(y_test_single_label, y_pred_single_label)

        # Get the number of unique classes
        num_classes = len(class_names)

        # Dynamically set the figure size based on the number of classes
        fig_size_width = max(10, num_classes * 1)
        fig_size_height = max(6.5, num_classes * 0.6)  # Slightly increased height

        # Create the plot with dynamic size
        fig, ax = plt.subplots(figsize=(fig_size_width, fig_size_height))
        image = ax.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix", fontsize=18)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, fontsize=14)
        plt.yticks(tick_marks, class_names, fontsize=14)

        # Add annotations
        thresh = conf.max() / 2.
        for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
            plt.text(j, i, format(conf[i, j], 'd'),
                     horizontalalignment="center",
                     fontsize=14,
                     color="white" if conf[i, j] > thresh else "black")

        # Adjust layout to prevent label cutoff
        plt.subplots_adjust(bottom=0.2)  # Further increase bottom margin
        plt.tight_layout(pad=2.0)  # Adjust layout with more padding
        plt.ylabel('True label', fontsize=16)
        plt.xlabel('Predicted label', fontsize=16)

        # Save the plot
        plot_filename = "plotGB.png"
        plt.savefig(plot_filename)
        plt.close()

        self.plot.setData(plot_filename)

        ############ TABLE WITH METRICS

        # Calculate accuracy, precision, recall, and f1 score
        accuracy2 = round(accuracy_score(y_test, y_pred) * 100, 1)
        precision2 = round(precision_score(y_test, y_pred, average='weighted') * 100, 1)
        recall2 = round(recall_score(y_test, y_pred, average='weighted') * 100, 1)
        f12 = round(f1_score(y_test, y_pred, average='weighted') * 100, 1)

        # Create the metrics table
        metrics_table = pd.DataFrame(
            {'Accuracy (%)': [accuracy2], 'Precision (%)': [precision2], 'Recall (%)': [recall2],
             'F1 Score (%)': [f12]})

        # Create figure and axis for the table
        fig, ax = plt.subplots(figsize=(8, 2))  # Adjust size as needed
        ax.axis('off')

        # Create a table and style it
        table = plt.table(cellText=metrics_table.values, colLabels=metrics_table.columns,
                          loc='center', cellLoc='center', edges='closed')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Adjust to suitable size

        # Style header cells
        for (i, col) in enumerate(metrics_table.columns):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data cells
        for i in range(1, len(metrics_table) + 1):
            for j in range(len(metrics_table.columns)):
                table[(i, j)].set_facecolor("#f1f1f2")

        plt.savefig('metrics_tableGB.png')
        plt.close()

        self.metrics_table.setData('metrics_tableGB.png')

        ############ CONFUSION TABLE

        # Convert class_metrics to a DataFrame
        confusion_table = pd.DataFrame(class_metrics).T

        # Create figure and axis for the table
        fig, ax = plt.subplots(figsize=(10, len(confusion_table) * 0.5))
        ax.axis('off')

        # Create a table and style it
        table = plt.table(cellText=confusion_table.values, colLabels=confusion_table.columns,
                          rowLabels=confusion_table.index, loc='center', cellLoc='center', edges='closed')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)  # Adjust to suitable size

        # Style header cells
        for (i, col) in enumerate(confusion_table.columns):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data cells
        for i in range(1, len(confusion_table) + 1):
            for j in range(len(confusion_table.columns)):
                table[(i, j)].set_facecolor("#f1f1f2")

        plt.savefig('confusion_tableGB.png')
        plt.close()

        self.confusion_table.setData('confusion_tableGB.png')

        ####################

        # Stop the timer and calculate the duration
        end_time = time.time()
        duration = end_time - start_time

        # Print the metrics and computation duration with labels
        # accuracy_percentage = acc * 100
        # print("The accuracy is: {:.2f}%".format(accuracy_percentage))
        print("The accuracy is {:.4f}".format(accuracy))
        print("The precision is {:.4f}".format(precision))
        print("The recall is {:.4f}".format(recall))
        print("The F1 Score is {:.4f}".format(f1))
        print("The True Positive are {}".format(int(true_positives)))
        print("The True Negative are {}".format(int(true_negatives)))
        print("The False Positive are {}".format(int(false_positives)))
        print("The False Negative are {}".format(int(false_negatives)))
        print("Computation duration is {:.4f} seconds".format(duration))

        # Show a message that the node has finished running
        finish_dialog = QMessageBox()
        finish_dialog.setWindowTitle("Information")
        finish_dialog.setText("Node successfully executed.")
        finish_dialog.setStandardButtons(QMessageBox.Ok)
        finish_dialog.exec_()

    def createUi(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        pin_labels = [
            ("x_train", self.x_train),
            ("x_train_", self.x_train_),
            ("y_train", self.y_train),
            ("x_test", self.x_test),
            ("x_test_", self.x_test_),
            ("y_test", self.y_test),
            ("accuracy", self.accuracy),
            ("f1", self.f1),
            ("trained_model", self.trained_model),
            ("metrics table", self.metrics_table),
            ("confusion table", self.confusion_table),
            ("plot", self.plot),
        ]

        for pin_name, pin in pin_labels:
            label = QtWidgets.QLabel(pin_name)
            layout.addWidget(label)

            value_label = QtWidgets.QLabel()
            layout.addWidget(value_label)

            def update_value_label(data):
                value_label.setText(str(data))

            pin.onPinDataChanged.connect(update_value_label)

        self.setWidget(widget)

    @staticmethod
    def category():
        return '3_Data_Classification'




