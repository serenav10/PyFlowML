from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from PySide2.QtWidgets import QLabel, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QPushButton
from PySide2.QtCore import Qt
from Qt import QtWidgets, QtCore
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
import graphviz
import random
from pycebox.ice import ice, ice_plot

# Individual Conditional Expectation (ICE) Plots

class ICE(NodeBase):
    def __init__(self, name):
        super(ICE, self).__init__(name)

        # Define the input and output pins
        self.trained_model = self.createInputPin('Trained Model', 'AnyPin')
        self.x_train = self.createInputPin('Data for Training', 'AnyPin')
        self.x_test = self.createInputPin('Data for Test', 'AnyPin')
        self.features_name = self.createInputPin('Name of Features', 'AnyPin')
        self.ice_values = self.createOutputPin("Ice Values", 'StringPin')

        # Enable the allowAny option for the input and output pins
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.trained_model.enableOptions(PinOptions.AllowAny)
        self.features_name.enableOptions(PinOptions.AllowAny)

        # Set default values for prompt
        self.features_name.setData('OverTime')

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addOutputDataType('StringPin')
        helper.addInputDataType('AnyPin')
        #helper.addInputDataType('StringPin')
        return helper

    def compute(self, *args, **kwargs):
        # Get the x_train, trained_model, and features_indices data from the input pins
        x_train = self.x_train.getData()
        trained_model = self.trained_model.getData()
        features_name = self.features_name.getData()

        # Convert the x_train data (NumPy array) to a DataFrame
        x_train_df = pd.DataFrame(x_train, columns=features_name)

        # Randomly select a subset of instances to generate ICE plots for
        num_instances_to_plot = 5  # You can change this number as needed
        selected_instances = random.sample(range(len(x_train_df)), num_instances_to_plot)

        # Randomly select a feature for ICE plots
        selected_feature = random.choice(features_name)
        feature_index = x_train_df.columns.get_loc(selected_feature)

        # Compute ICE values for the selected feature and instances
        ice_values = ice(trained_model, x_train_df, feature_index)

        # Plot the ICE plots
        plt.figure(figsize=(20, 16))
        ice_plot(ice_values, c="steelblue", alpha=0.4, linewidth=0.5)
        plt.xlabel(selected_feature)
        plt.ylabel('Model Prediction')
        plt.title('ICE Plots for Selected Instances')
        plt.savefig('ICE_Plots.png')
        plt.close()

        # Set the ICE plot file path for the output pin
        self.ice_values.setData('ICE_Plots.png')

    def createUi(self):
        # Create the main widget for the custom UI
        widget = QtWidgets.QWidget()

        # Create a layout for the widget
        layout = QtWidgets.QVBoxLayout(widget)

        # Create a function to display the ICE plot
        def display_ice_plot(data):
            if isinstance(data, str) and data.endswith('.png'):
                pixmap = QtGui.QPixmap(data)
                value_label.setPixmap(pixmap)

        self.ice_values.onPinDataChanged.connect(display_ice_plot)

        # Create a function to display the ICE plot
        def display_ice_plot(data):
            if isinstance(data, str) and data.endswith('.png'):
                pixmap = QtGui.QPixmap(data)
                value_label.setPixmap(pixmap)

                # Assuming features_names is a comma-separated string
                feature_labels = self.features_name.getData().split(',')

                # Set custom x-axis tick labels using feature names
                plt.xticks(range(len(feature_labels)), feature_labels, rotation='vertical')

        self.ice_values.onPinDataChanged.connect(display_ice_plot)

        # Create labels to display the pin names and values
        pin_labels = [
            ("x_train", self.x_train),
            ("x_test", self.x_test),
            ("ice_plot", self.ice_values),  # Update the label to reflect ICE plot
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
        return '4_Explainable_AI'


