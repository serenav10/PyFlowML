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

# Partial Dependence Plot (PDP) is a graphical tool used to understand the relationship between a set of input
# features (variables) and the predicted outcome of a machine learning model. It helps to visualize how the predicted
# outcome changes as one or more features vary while holding other features constant

class Partial_Dependence_Plot(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False

        ####################### prompt only on refreshing this node
        async def refresh_node(node):
            # Check if the node is of type 'PDP'
            if node.type == 'Partial_Dependence_Plot':
                # Show the prompt for 'PDP'
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')

        #######################
        super(Partial_Dependence_Plot, self).__init__(name)

        # Define the input and output pins
        self.trained_model = self.createInputPin('Trained Model', 'AnyPin')
        self.x_train = self.createInputPin('Data for Training', 'AnyPin')
        self.x_test = self.createInputPin('Data for Test', 'AnyPin')
        self.features_name = self.createInputPin('Name od the Features', 'AnyPin')
        self.pdp_values = self.createOutputPin('Partial_Dependence_values', 'StringPin')

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

        # Randomly select a subset of features
        num_features_to_plot = 12  # You can change this number as needed
        selected_features = random.sample(features_name, num_features_to_plot)

        # Find the indices of the selected features in the dataset
        features_indices = [x_train_df.columns.get_loc(feature_name) for feature_name in selected_features]

        # Generate PDP plots using PartialDependenceDisplay for the selected features
        pdp_display = PartialDependenceDisplay.from_estimator(
            trained_model, x_train_df, features_indices)

        # Plot the PDP
        plt.figure(figsize=(20, 16))
        pdp_display.plot(ax=plt.gca())
        plt.savefig('Partial_Dependence_Plot.png')
        plt.close()

        # Set the PDP plot file path for the output pin
        self.pdp_values.setData('Partial_Dependence_Plot.png')

    def createUi(self):
        # Create the main widget for the custom UI
        widget = QtWidgets.QWidget()

        # Create a layout for the widget
        layout = QtWidgets.QVBoxLayout(widget)

        # Create a function to display the PDP plot
        def display_pdp_plot(data):
            if isinstance(data, str) and data.endswith('.png'):
                pixmap = QtGui.QPixmap(data)
                value_label.setPixmap(pixmap)

        self.pdp_values.onPinDataChanged.connect(display_pdp_plot)

        # Create a function to display the PDP plot
        def display_pdp_plot(data):
            if isinstance(data, str) and data.endswith('.png'):
                pixmap = QtGui.QPixmap(data)
                value_label.setPixmap(pixmap)

                # Assuming features_names is a comma-separated string
                feature_labels = self.features_name.getData().split(',')

                # Set custom x-axis tick labels using feature names
                plt.xticks(range(len(feature_labels)), feature_labels, rotation='vertical')

        self.pdp_values.onPinDataChanged.connect(display_pdp_plot)

        # Create labels to display the pin names and values
        pin_labels = [
            ("x_train", self.x_train),
            ("x_test", self.x_test),
            ("pdp_plot", self.pdp_values),  # Update the label to reflect PDP plot
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


