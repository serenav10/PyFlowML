from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
import tensorflow as tf
from tensorflow import keras
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import shap
import warnings
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from PySide2.QtWidgets import QLabel, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QPushButton, QMessageBox
from PySide2.QtCore import Qt
from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets, QtCore

# INSIGHTS: https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability#

class SHAP(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False
        # self.parametersSet = False
        self.messagesShown = False

        ####################### prompt only on refreshing this node
        async def refresh_node(node):
            if node.type == 'SHAP':
                # Show the prompt
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')

        #######################

        super(SHAP, self).__init__(name)

        # Define the input and output pins
        self.trained_model = self.createInputPin('Trained Model', 'AnyPin')
        self.x_train = self.createInputPin("Data for Training", 'AnyPin')
        self.x_test = self.createInputPin("Data for Test", 'AnyPin')
        self.features_name = self.createInputPin('Name of Features', 'AnyPin')
        self.shap_values = self.createOutputPin("Shapley Values Summary Plot", 'StringPin')
        self.force_plot = self.createOutputPin("Shapley Values Force Plot", 'StringPin')
        self.decision_plot = self.createOutputPin("Shapley Values Decision Plot", 'StringPin')

        # Enable the allowAny option for the input and output pins
        self.x_train.enableOptions(PinOptions.AllowAny)
        self.x_test.enableOptions(PinOptions.AllowAny)
        self.trained_model.enableOptions(PinOptions.AllowAny)
        self.shap_values.enableOptions(PinOptions.AllowAny)
        self.features_name.enableOptions(PinOptions.AllowAny)

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addOutputDataType('StringPin')
        helper.addInputDataType('AnyPin')
        helper.addInputDataType('StringPin')
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
        plt.ioff()
        self.promptVariables()

        # Get the x_train, y_train, x_test, and y_test data from the input pins
        x_train = self.x_train.getData()
        x_test = self.x_test.getData()
        trained_model = self.trained_model.getData()
        features_name = self.features_name.getData()
        print("Feature names:", features_name)

        # Initialize plot paths
        summary_plot_path = 'ShapValues.png'
        force_plot_path = 'ForcePlot.png'
        decision_plot_path = 'DecisionPlot.png'

        # Create a DataFrame with the original feature names as column names
        df_x_test = pd.DataFrame(x_test, columns=features_name)

        # Check the type of the trained model and create the appropriate SHAP explainer
        if isinstance(trained_model, MLPClassifier):
            # Use the original approach for MLPClassifier
            wrapped_model = lambda x: trained_model.predict_proba(x)[:, 1]
            explainer = shap.Explainer(wrapped_model, x_train)
            shap_values = explainer(df_x_test)

            # Create the summary plot for MLPClassifier
            plt.figure(figsize=(12, 10))  # Adjust the figure size
            shap.summary_plot(shap_values, df_x_test, feature_names=features_name, show=False)

            # Add title to the plot
            if hasattr(trained_model, 'classes_'):
                class_label = str(trained_model.classes_[1])  # Assuming the second class is the positive class
            else:
                class_label = "Positive Class"
            plt.title(f'SHAP Summary Plot for target variable class: {class_label}')

            summary_plot_path = 'ShapValues_MLP.png'
            plt.tight_layout()
            plt.savefig(summary_plot_path)
            plt.close()

            # Use KernelExplainer specifically for force plot and decision plot for MLPClassifier
            kernel_explainer = shap.KernelExplainer(wrapped_model, shap.utils.sample(x_train, 100))
            # Calculate SHAP values for the first sample
            kernel_shap_values = kernel_explainer.shap_values(df_x_test.iloc[0, :])

            # Create a copy of df_x_test and round the values to 3 decimal places
            df_x_test_rounded = df_x_test.round(3)

            try:
                # Generate and save the force plot for the first sample using KernelExplainer
                force_plot_path = 'ForcePlot_MLP.png'
                plt.figure(figsize=(20, 8))

                shap.force_plot(kernel_explainer.expected_value, kernel_shap_values, df_x_test_rounded.iloc[0, :], matplotlib=True,
                                show=False, text_rotation=15, figsize=(20, 8))
                plt.savefig(force_plot_path, dpi=100, bbox_inches="tight", format='png')
                plt.close()
            except Exception as e:
                print("Error generating decision plot:", str(e))
                force_plot_path = "Work in progress"

            # Define the number of samples you want to generate SHAP values for
            n_samples = 10

            # Calculate SHAP values for multiple samples
            shap_values_multiple_samples = kernel_explainer.shap_values(df_x_test.iloc[0:n_samples, :])

            try:
                # Generate and save the decision plot for multiple samples
                decision_plot_path = 'DecisionPlot_MLP_Multiple.png'
                #plt.figure(figsize=(20, 8))
                shap.decision_plot(kernel_explainer.expected_value, shap_values_multiple_samples,
                                   df_x_test.iloc[0:n_samples, :], show=False)
                plt.title("Decision Plot (10 samples)")
                plt.tight_layout()
                plt.savefig(decision_plot_path, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print("Error generating decision plot:", str(e))
                force_plot_path = "Work in progress"


        elif isinstance(trained_model, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)):
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(trained_model)
            shap_values = explainer.shap_values(df_x_test)

            # Check if the model is Gradient Boosting Classifier
            if isinstance(trained_model, GradientBoostingClassifier):
                # Handle SHAP values specifically for Gradient Boosting Classifier
                # Check the structure of SHAP values and use appropriate indexing
                shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

            else:

                # For other tree-based models, use the SHAP values for class "1" (positive class)
                shap_values_to_plot = shap_values[1]

            # Get class label for class "1" if available
            class_label = trained_model.classes_[1] if hasattr(trained_model, 'classes_') else "Class 1"

            # Increase the size of the figure
            plt.figure(figsize=(10, 8))  # Adjust the figure size
            # Create the summary plot for tree-based models
            shap.summary_plot(shap_values_to_plot, df_x_test, feature_names=features_name, show=False)
            # Update plot title to include class label
            plt.title(f'SHAP Summary Plot for target variable class: {class_label}')

            # Shorten the x-axis label
            plt.xlabel('Mean |SHAP Value|')
            summary_plot_path = 'ShapValues_TreeModels.png'
            plt.tight_layout()
            plt.savefig(summary_plot_path)
            plt.close()

            # Create a copy of df_x_test and round the values to 3 decimal places
            df_x_test_rounded = df_x_test.round(3)

            # Generate and save the force plot for the first sample
            if isinstance(trained_model, GradientBoostingClassifier):
                # For Gradient Boosting, if explainer.expected_value is just a single number (float),
                # use it directly. Otherwise, try to access the second element for the positive class.
                expected_value = explainer.expected_value
                if isinstance(expected_value, np.ndarray) and expected_value.shape[0] > 1:
                    expected_value = expected_value[1]
                shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
            else:
                # For other tree-based models, like Decision Tree and Random Forest
                # Check if explainer.expected_value is an array and has more than one element
                if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) > 1:
                    expected_value = explainer.expected_value[1]  # for positive class
                else:
                    expected_value = explainer.expected_value  # for regression or single-output models
                shap_values_to_plot = shap_values[1]

            try:
                # Existing force plot code 
                force_plot_path = 'ForcePlot_TreeModels.png'
                plt.figure(figsize=(20, 8))

                shap.force_plot(kernel_explainer.expected_value, kernel_shap_values, df_x_test_rounded.iloc[0, :],
                                matplotlib=True,
                                show=False, text_rotation=15, figsize=(20, 8))
                plt.savefig(force_plot_path, dpi=100, bbox_inches="tight", format='png')
                plt.close()
            except Exception as e:
                print("Error generating force plot:", str(e))
                force_plot_path = "Work in progress"

            # Generate and save the decision plot for the first sample
            if isinstance(trained_model, GradientBoostingClassifier):
                # For Gradient Boosting, use the expected value directly if it's a single number
                # If it's an array and has more than one value (e.g., for multi-class classification), then use the second value
                expected_value = explainer.expected_value
                if isinstance(expected_value, np.ndarray) and expected_value.shape[0] > 1:
                    expected_value = expected_value[1]
            else:
                # For other models, check if explainer.expected_value is an array and has more than one element
                expected_value = explainer.expected_value
                if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) > 1:
                    expected_value = explainer.expected_value[1]

            try:
                decision_plot_path = 'DecisionPlot_TreeModels.png'
                plt.figure(figsize=(20, 8))
                plt.title("Decision Plot")
                shap.decision_plot(expected_value, shap_values_to_plot, df_x_test.iloc[0, :], show=False)
                plt.tight_layout()
                plt.savefig(decision_plot_path, bbox_inches="tight")
                plt.close()

            except Exception as e:
                print("Error generating decision plot:", str(e))
                force_plot_path = "Work in progress"

        else:
            # For other models, use the high-level SHAP explainer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                explainer = shap.Explainer(trained_model, x_train)
                shap_values = explainer(df_x_test)

            # Create the summary plot for other models
            shap.summary_plot(shap_values, df_x_test, feature_names=features_name, show=False)
            summary_plot_path = 'ShapValues_OtherModels.png'
            plt.tight_layout()
            plt.savefig(summary_plot_path)

            force_plot_path = 'ForcePlot_OtherModels.png'
            decision_plot_path = 'DecisionPlot_OtherModels.png'


        plt.close()

        # Set the SHAP values for the output pin as the file path of the saved plot
        self.shap_values.setData(summary_plot_path)
        self.force_plot.setData(force_plot_path)
        self.decision_plot.setData(decision_plot_path)

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
            ("x_test", self.x_test),
            ("shap_values", self.shap_values),
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
