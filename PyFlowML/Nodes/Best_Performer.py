from PyFlow.Core import NodeBase
from PyFlow.Core.Common import *
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets, QtCore
import numpy as np
import pandas as pd
from PySide2.QtWidgets import QLabel, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QPushButton, QMessageBox
from PySide2.QtCore import Qt
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from matplotlib.table import table
from collections import Counter

class Best_Performer(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False
        #self.parametersSet = False
        self.messagesShown = False

        ####################### prompt only on refreshing this node
        async def refresh_node(node):
            if node.type == 'Best_Performer':
                # Show the prompt
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')

        #######################

        super(Best_Performer, self).__init__(name)

        self.Decision_Tree_Accuracy = self.createInputPin("Decision_Tree_Accuracy", 'FloatPin')
        self.Decision_Tree_F1 = self.createInputPin("Decision_Tree_F1", 'FloatPin')

        self.Deep_Neural_Network_Accuracy = self.createInputPin("Deep_Neural_Network_Accuracy", 'FloatPin')
        self.Deep_Neural_Network_F1 = self.createInputPin("Deep_Neural_Network_F1", 'FloatPin')

        self.Support_Vector_Machine_Accuracy = self.createInputPin("Support_Vector_Machine_Accuracy", 'FloatPin')
        self.Support_Vector_Machine_F1 = self.createInputPin("Support_Vector_Machine_F1", 'FloatPin')

        self.Multinomial_Naive_Bayes_Accuracy = self.createInputPin("Multinomial_Naive_Bayes_Accuracy", 'FloatPin')
        self.Multinomial_Naive_Bayes_F1 = self.createInputPin("Multinomial_Naive_Bayes_F1", 'FloatPin')

        self.Random_Forest_Accuracy = self.createInputPin("Random_Forest_Accuracy", 'FloatPin')
        self.Random_Forest_F1 = self.createInputPin("Random_Forest_F1", 'FloatPin')

        self.Gradient_Boosting_Accuracy = self.createInputPin("Gradient_Boosting_Accuracy", 'FloatPin')
        self.Gradient_Boosting_F1 = self.createInputPin("Gradient_Boosting_F1", 'FloatPin')

        self.K_Nearest_Neighbors_Accuracy = self.createInputPin("K_Nearest_Neighbors_Accuracy", 'FloatPin')
        self.K_Nearest_Neighbors_F1 = self.createInputPin("K_Nearest_Neighbors_F1", 'FloatPin')

        self.BERT_Accuracy = self.createInputPin("BERT_Accuracy", 'FloatPin')
        self.BERT_F1 = self.createInputPin("BERT_F1", 'FloatPin')

        self.Best_Performer = self.createOutputPin("Best_Performer", 'StringPin')


    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('IntPin')
        helper.addInputDataType('FloatPin')
        helper.addOutputDataType('FloatPin')
        helper.addOutputDataType('IntPin')
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

        max_accuracy = 0
        max_accuracy_pin_name = ""
        max_f1 = 0
        max_f1_pin_name = ""

        # Define the mapping dictionary
        mapping_dict = {
            "Decision_Tree": "Decision Tree",
            "BERT": "BERT",
            "K_Nearest_Neighbor": "K-nearest neighbor",
            "Deep_Neural_Network": "Deep Neural Network",
            "Support_Vector_Machine": "Support Vector Machine",
            "Multinomial_Naive_Bayes": "Multinomial Naive Bayes",
            "Random_Forest": "Random Forest",
            "Gradient_Boosting": "Gradient Boosting"
        }

        # Find the highest accuracy value and its corresponding pin name and classifier
        for pin in self.inputs.values():
            if pin.name.endswith("_Accuracy"):
                value = pin.getData()
                if value is not None and round(value, 3) > max_accuracy:
                    max_accuracy = round(value, 3)
                    max_accuracy_pin_name = pin.name
                    max_accuracy_classifier = "_".join(pin.name.split("_")[:-1])  # Extract the classifier name

        # Find the highest F1 score value and its corresponding pin name and classifier
        for pin in self.inputs.values():
            if pin.name.endswith("_F1"):
                value = pin.getData()
                if value is not None and round(value, 3) > max_f1:
                    max_f1 = round(value, 3)
                    max_f1_pin_name = pin.name
                    max_f1_classifier = "_".join(pin.name.split("_")[:-1])  # Extract the classifier name

        # Find the highest accuracy value and its corresponding pin name and classifier
        for pin in self.inputs.values():
            if pin.name.endswith("_Accuracy"):
                value = pin.getData()
                if value is not None and round(value, 3) > max_accuracy:
                    max_accuracy = round(value, 3)
                    max_accuracy_pin_name = pin.name
                    max_accuracy_classifier = "_".join(pin.name.split("_")[:-1])  # Extract the classifier name

        # Find the highest F1 score value and its corresponding pin name and classifier
        for pin in self.inputs.values():
            if pin.name.endswith("_F1"):
                value = pin.getData()
                if value is not None and round(value, 3) > max_f1:
                    max_f1 = round(value, 3)
                    max_f1_pin_name = pin.name
                    max_f1_classifier = "_".join(pin.name.split("_")[:-1])  # Extract the classifier name

        # Map the classifier names to more readable names using mapping_dict
        max_accuracy_readable = mapping_dict.get(max_accuracy_classifier, max_accuracy_classifier)
        max_f1_readable = mapping_dict.get(max_f1_classifier, max_f1_classifier)

        # Assuming max_accuracy and max_f1 are defined
        max_accuracy_percent = round(max_accuracy * 100, 1)
        max_f1_percent = round(max_f1 * 100, 1)

        # DataFrame for the table
        bestperf = pd.DataFrame([
            {'Best Performer': max_accuracy_readable, 'Highest Accuracy (%)': max_accuracy_percent,
             'Highest F1 score (%)': ' '},
            {'Best Performer': max_f1_readable, 'Highest Accuracy (%)': ' ', 'Highest F1 score (%)': max_f1_percent}
        ])

        # Create figure and axis for the table
        fig, ax = plt.subplots(figsize=(10, 4))  # Adjust size as needed
        ax.axis('off')

        # Create a table and style it
        table = plt.table(cellText=bestperf.values, colLabels=bestperf.columns,
                          loc='center', cellLoc='center', edges='closed')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Adjust to suitable size

        # Style header cells
        for (i, col) in enumerate(bestperf.columns):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data cells
        for i in range(1, len(bestperf) + 1):
            for j in range(len(bestperf.columns)):
                table[(i, j)].set_facecolor("#f1f1f2")

        plt.savefig('best_performer_styled.png')
        plt.close()

        self.Best_Performer.setData('best_performer_styled.png')

        # Show a message that the node has finished running
        finish_dialog = QMessageBox()
        finish_dialog.setWindowTitle("Information")
        finish_dialog.setText("Node successfully executed.")
        finish_dialog.setStandardButtons(QMessageBox.Ok)
        finish_dialog.exec_()

    def get_previous_node_name(self, pin_name):
        previous_node = None
        for node in self.graph().getNodes():
            if node is self:
                break
            if isinstance(node, NodeBase):
                for pin in node.outputs.values():
                    if pin.isPinName(pin_name):
                        previous_node = node
                        break
        if previous_node is not None:
            return previous_node.getName()
        else:
            return "Unknown"

    @staticmethod
    def category():
        return '2_Data_Visualization'
