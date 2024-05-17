from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from PyFlow.Core.Common import *
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Packages.PyFlowBase.Pins.BoolPin import BoolPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets
import numpy as np
import time
from sklearn.naive_bayes import MultinomialNB
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

class Multinomial_Naive_Bayes(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False
        # self.parametersSet = False
        self.messagesShown = False

        ####################### prompt only on refreshing this node
        async def refresh_node(node):
            if node.type == 'Multinomial_Naive_Bayes':
                # Show the prompt
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')

        #######################

        super(Multinomial_Naive_Bayes, self).__init__(name)
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

        # Get the data from input pins
        x_train = self.x_train.getData()
        y_train = self.y_train.getData()
        x_test = self.x_test.getData()
        y_test = self.y_test.getData()
        class_names = self.classes_name.getData()

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

        # Create and train the Naive Bayes classifier
        clf = MultinomialNB()
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
        fig_size_width = max(8, num_classes)  # Minimum width of 8, increases with more classes
        fig_size_height = max(6, num_classes * 0.5)  # Minimum height of 6, scales with the number of classes

        # Create the plot with dynamic size
        fig, ax = plt.subplots(figsize=(fig_size_width, fig_size_height))
        image = ax.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix", fontsize=14)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Add annotations
        thresh = conf.max() / 2.
        for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
            plt.text(j, i, format(conf[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)

        # Save the plot
        plot_filename = "plotNB.png"
        plt.savefig(plot_filename)
        plt.close()

        self.plot.setData(plot_filename)

        ############ TABLE WITH METRICS

        # Calculate accuracy, precision, recall, and f1 score
        accuracy = round(accuracy_score(y_test, y_pred), 4)
        precision = round(precision_score(y_test, y_pred, average='weighted'), 4)
        recall = round(recall_score(y_test, y_pred, average='weighted'), 4)
        f1 = round(f1_score(y_test, y_pred, average='weighted'), 4)

        # Create and save the metrics table as an image
        metrics_table = pd.DataFrame(
            {'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1 Score': [f1]})

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(8, 4))

        # Create the table and adjust font size and cell padding
        table = plt.table(cellText=metrics_table.values, colLabels=metrics_table.columns, cellLoc='center',
                          loc='center', cellColours=plt.cm.BuPu(np.full_like(metrics_table.values, 0.1)))

        # Adjust font size
        table.set_fontsize(12)

        # Adjust cell padding
        table.scale(1, 1.5)

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('metrics_tableNB.png')
        plt.close()

        self.metrics_table.setData('metrics_tableNB.png')

        ############ CONFUSION TABLE

        # Convert class_metrics to a DataFrame
        confusion_table = pd.DataFrame(class_metrics).T  # Transpose to have classes as rows

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(10, len(confusion_table) * 0.5))  # Adjust the height based on the number of classes

        # Create the table and adjust font size and cell padding
        table2 = plt.table(cellText=confusion_table.values, colLabels=confusion_table.columns,
                           rowLabels=confusion_table.index,
                           cellLoc='center', loc='center', cellColours=plt.cm.BuPu(np.full(confusion_table.shape, 0.1)))

        # Adjust font size
        table2.set_fontsize(12)

        # Adjust cell padding
        table2.scale(1, 1.5)

        # Remove axis and spines
        plt.axis('off')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.savefig('confusion_tableNB.png')
        plt.close()

        self.confusion_table.setData('confusion_tableNB.png')

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



