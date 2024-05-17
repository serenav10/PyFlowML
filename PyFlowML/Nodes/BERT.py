from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from PyFlow.Core.Common import *
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Packages.PyFlowBase.Pins.BoolPin import BoolPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets
import numpy as np
import time
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List
from matplotlib.table import table
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import itertools



# BERT Classifier Node
class BERT(NodeBase):
    def __init__(self, name):
        super(BERT, self).__init__(name)

        # Define the input pins
        self.x_train = self.createInputPin("x_train", 'AnyPin')
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.y_train = self.createInputPin("y_train", 'AnyPin')
        self.y_train.enableOptions(PinOptions.AllowAny)
        self.x_test = self.createInputPin("x_test", 'AnyPin')
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.y_test = self.createInputPin("y_test", 'AnyPin')
        self.y_test.enableOptions(PinOptions.AllowAny)
        # self.num_classes = self.createInputPin('num_classes', 'IntPin')

        # Define the output pins
        self.accuracy = self.createOutputPin("accuracy", 'FloatPin')
        # self.precision = self.createOutputPin("precision", 'FloatPin')
        # self.recall = self.createOutputPin("recall", 'FloatPin')
        self.f1 = self.createOutputPin("f1", 'FloatPin')
        # self.true_positives = self.createOutputPin("true_positives", 'IntPin')
        # self.true_negatives = self.createOutputPin("true_negatives", 'IntPin')
        # self.false_positives = self.createOutputPin("false_positives", 'IntPin')
        # self.false_negatives = self.createOutputPin("false_negatives", 'IntPin')
        self.confusion_table = self.createOutputPin('confusion_table', 'StringPin')
        self.plot = self.createOutputPin('plot', 'StringPin')
        self.metrics_table = self.createOutputPin('metrics_table', 'StringPin')


    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('AnyPin')
        helper.addOutputDataType('FloatPin')
        helper.addInputDataType('IntPin')
        helper.addOutputDataType('AnyPin')
        helper.addOutputDataType('StringPin')
        return helper

    def compute(self, *args, **kwargs):
        # Get the data from input pins
        x_train = self.x_train.getData()
        y_train = self.y_train.getData()
        x_test = self.x_test.getData()
        y_test = self.y_test.getData()
        # num_classes = self.num_classes.getData()

        # Convert y_train to a NumPy array
        y_train = np.array(y_train)

        # Reshape y_train to be a 1D array
        # y_train = np.argmax(y_train, axis=1)
        # If y_train is one-hot encoded, convert it back to the original format
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)

        # Convert y_test to a NumPy array and reshape if necessary
        y_test = np.array(y_test)
        # y_test = np.argmax(y_test, axis=1)
        # If y_test is one-hot encoded, convert it back to the original format
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)

        x_train = [str(x) for x in x_train]
        x_test = [str(x) for x in x_test]

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        x_train_tokenized = tokenizer.batch_encode_plus(
            x_train,
            padding=True,
            truncation=True,
            return_tensors='tf'
        )
        x_test_tokenized = tokenizer.batch_encode_plus(
            x_test,
            padding=True,
            truncation=True,
            return_tensors='tf'
        )

        # Convert tokenized data to TensorFlow Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(x_train_tokenized),
            y_train
        )).batch(16)
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(x_test_tokenized),
            y_test
        )).batch(16)

        # Start the timer
        start_time = time.time()

        # Create BERT model
        num_classes = 2 # Specify the number of classes in your classification task
        bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

        # Define optimizer and loss function
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

        # Compile the model
        bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        bert_model.fit(train_dataset, epochs=2, batch_size = 128)

        eval_results = bert_model.evaluate(test_dataset)
        accuracy = eval_results[1]

        # Predict on the test set
        y_pred = tf.argmax(bert_model.predict(test_dataset), axis=1)

        # Calculate precision, recall, and F1 score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Set the data for the output pins
        self.accuracy.setData(accuracy)
        self.precision.setData(precision)
        self.recall.setData(recall)
        self.f1.setData(f1)

        # Set the accuracy, precision, recall, and f1 score as outputs
        self.accuracy.setData(accuracy)
        # self.precision.setData(precision)
        # self.recall.setData(recall)
        self.f1.setData(f1)

        # Convert y_test and y_pred to a NumPy array
        y_test_single_label = np.array(y_test)
        y_pred_single_label = np.array(y_pred)

        # If y_test and y_pred are one-hot encoded, convert them back to the original format
        if len(y_test_single_label.shape) > 1:
            y_test_single_label = np.argmax(y_test, axis=1)
        if len(y_pred_single_label.shape) > 1:
            y_pred_single_label = np.argmax(y_pred_single_label, axis=1)

        # Generate the confusion matrix
        conf = metrics.confusion_matrix(y_test_single_label, y_pred_single_label)

        # Calculate true positives, true negatives, false positives, and false negatives
        true_positives = np.sum((y_pred_single_label == 1) & (y_test_single_label == 1))
        true_negatives = np.sum((y_pred_single_label == 0) & (y_test_single_label == 0))
        false_positives = np.sum((y_pred_single_label == 1) & (y_test_single_label == 0))
        false_negatives = np.sum((y_pred_single_label == 0) & (y_test_single_label == 1))

        ################ CONFUSION MATRIX

        # Define the class labels
        classes = [0, 1]

        # Compute the minimum and maximum values in the data
        # data_min = np.min(conf)
        data_max = np.max(conf)

        # Create the plot
        fig, ax = plt.subplots()
        image = ax.imshow(conf, interpolation='bilinear', cmap=plt.cm.GnBu,
                          extent=[-0.5, len(classes) - 0.5, -0.5, len(classes) - 0.5], vmin=0, vmax=conf.max())
        plt.title("Confusion Matrix", fontsize=12)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = conf.max() / 2.
        for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
            plt.text(j, i, format(conf[i, j], fmt),
                     fontsize=13,  # Adjust the font size as desired
                     horizontalalignment="center",
                     va="top",
                     color="black")

        # Add labels to the squares
        label_00 = 'True Negative'
        label_01 = 'False Positive'
        label_10 = 'False Negative'
        label_11 = 'True Positive'

        text = ax.text(0, 0, label_00, ha="center", va="bottom", color="black", fontsize=12)
        text = ax.text(1, 0, label_01, ha="center", va="bottom", color="black", fontsize=12)
        text = ax.text(0, 1, label_10, ha="center", va="bottom", color="black", fontsize=12)
        text = ax.text(1, 1, label_11, ha="center", va="bottom", color="black", fontsize=12)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=12, labelpad=-2)
        plt.xlabel('Predicted label', fontsize=12, labelpad=-2)

        # Set the data for the output pins
        # self.confusion_matrix.setData(conf)
        # Save the plot as a PNG image
        plot_filenameBERT = "plotBERT.png"
        plt.savefig(plot_filenameBERT)

        # Set the plot filename as the data for the output pin
        self.plot.setData(plot_filenameBERT)

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

        plt.savefig('metrics_tableBERT.png')
        plt.close()

        self.metrics_table.setData('metrics_tableBERT.png')

        ############ CONFUSION TABLE

        # Create and save the metrics table as an image
        confusion_table = pd.DataFrame(
            {'True Positives': [true_positives], 'True Negatives': [true_negatives],
             'False Positives': [false_positives], 'False Negatives': [false_negatives]})

        # Increase the figure size to accommodate the table
        plt.figure(figsize=(8, 4))

        # Create the table and adjust font size and cell padding
        table2 = plt.table(cellText=confusion_table.values, colLabels=confusion_table.columns, cellLoc='center',
                           loc='center', cellColours=plt.cm.BuPu(np.full_like(confusion_table.values, 0.1)))

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

        plt.savefig('confusion_tableBERT.png')
        plt.close()

        self.confusion_table.setData('confusion_tableBERT.png')

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

    def createUi(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        pin_labels = [
            ("x_train", self.x_train),
            ("y_train", self.y_train),
            ("x_test", self.x_test),
            ("y_test", self.y_test),
            # ("num_classes", self.num_classes),
            ("accuracy", self.accuracy),
            # ("precision", self.precision),
            ("f1", self.f1),
            # ("recall", self.recall),
            # ("true positive", self.true_positives),
            # ("true negative", self.true_negatives),
            # ("false positive", self.false_positives),
            # ("false negative", self.false_negatives),
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





