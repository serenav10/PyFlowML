from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets, QtCore
from sklearn.tree import DecisionTreeClassifier, plot_tree
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import graphviz
from sklearn.tree import export_text, _tree

# LIME (Local Interpretable Model-agnostic Explanations): LIME is another widely used library for explaining black-box
# machine learning models. It generates locally interpretable explanations by fitting an interpretable model around the prediction

class LIME(NodeBase):
    def __init__(self, name):
        super(LIME, self).__init__(name)

        # Define the input and output pins
        self.trained_model = self.createInputPin('Trained Dataset', 'AnyPin')
        self.x_train = self.createInputPin('Data for Training', 'AnyPin')
        self.x_test = self.createInputPin('Data for Test', 'AnyPin')
        self.features_name = self.createInputPin('Name of Features', 'AnyPin')
        self.lime_values = self.createOutputPin("Lime Values", 'StringPin')

        # Enable the allowAny option for the input and output pins
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.trained_model.enableOptions(PinOptions.AllowAny)
        self.features_name.enableOptions(PinOptions.AllowAny)

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addOutputDataType('StringPin')
        helper.addInputDataType('AnyPin')
        return helper

    def compute(self, *args, **kwargs):
        # Get the x_train, y_train, x_test, and y_test data from the input pins
        x_train = self.x_train.getData()
        x_test = self.x_test.getData()
        trained_model = self.trained_model.getData()
        features_name = self.features_name.getData()
        print("Feature names:", features_name)

        # Create a DataFrame with the original feature names as column names
        df_x_test = pd.DataFrame(x_test, columns=features_name)

        # Create a LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=features_name, class_names=['0', '1'],
                                                           discretize_continuous=True)

        # Generate LIME explanations for each instance in x_test
        lime_explanations = []
        for i in range(len(x_test)):
            exp = explainer.explain_instance(x_test[i], trained_model.predict_proba, num_features=len(features_name))
            lime_explanations.append(exp.as_list())

        # Set the LIME explanations for the output pin
        self.lime_values.setData(lime_explanations)

        # Extract feature names and weights for each instance
        instance_names = ['Instance ' + str(i) for i in range(len(x_test))]
        feature_names = features_name
        weights = np.array(lime_explanations)

        # Create a figure for the bar plots
        plt.figure(figsize=(12, 8))

        # Set the number of instances to show in the bar plots
        max_instances_to_show = 20  # You can adjust this number

        # Select a subset of instances for the bar plots
        selected_instances = np.random.choice(len(x_test), size=max_instances_to_show, replace=False)

        # Iterate over selected instances and their corresponding LIME explanations
        for instance_idx in selected_instances:
            instance_name = instance_names[instance_idx]
            lime_exp = lime_explanations[instance_idx]
            feature_names = [name for name, _ in lime_exp]
            feature_weights = [weight for _, weight in lime_exp]

            # Create a figure for the bar plots with a larger figsize
            plt.figure(figsize=(15, 10))  # Adjust the values (width, height) as needed

            # Sort feature names and weights in descending order of weights
            sorted_features = [f for _, f in sorted(zip(feature_weights, feature_names), reverse=True)]
            sorted_weights = sorted(feature_weights, reverse=True)

            # Create a bar plot for feature weights with the sorted order
            plt.figure(figsize=(15, 10))  # Adjust the values (width, height) as needed
            plt.barh(sorted_features, sorted_weights, color='darkgreen')
            plt.xlabel('Feature Weight')
            plt.ylabel('Feature Name')
            plt.title(f'LIME Feature Weights for {instance_name}')
            plt.tight_layout()

            # Save the bar plot for the instance
            barplot_path = f'LIME_Feature_Weights_{instance_name}.png'
            plt.savefig(barplot_path)
            plt.close()

            # Set the bar plot file path for the output pin
            self.lime_values.setData(barplot_path)

    def createUi(self):
        # Create the main widget for the custom UI
        widget = QtWidgets.QWidget()

        # Create a layout for the widget
        layout = QtWidgets.QVBoxLayout(widget)

        # Create a function to display the LIME explanation plot
        def display_lime_explanations(data):
            if isinstance(data, str) and data.endswith('.png'):
                pixmap = QtGui.QPixmap(data)
                value_label.setPixmap(pixmap)

        self.lime_values.onPinDataChanged.connect(display_lime_explanations)

        # Create labels to display the pin names and values
        pin_labels = [
            ("x_train", self.x_train),
            #("y_train", self.y_train),
            ("x_test", self.x_test),
            #("y_test", self.y_test),
            ("lime_values", self.lime_values),
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


