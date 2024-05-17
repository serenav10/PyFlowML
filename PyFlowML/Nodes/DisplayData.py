from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from tensorflow import keras
from Qt import QtWidgets, QtCore
import numpy as np
from sklearn.datasets import load_breast_cancer
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets

##### THIS NODE DISPLAY ANY PIN IN  LOG #### USEFUL FOR MANAGING DATA AND CHECKING EXAMPLES####
class DisplayData(NodeBase):
    def __init__(self, name):
        super(DisplayData, self).__init__(name)

        # Define the input pin
        self.view = self.createInputPin('view', 'AnyPin')

        # Enable the allowAny option for the input pin
        self.view.enableOptions(PinOptions.AllowAny)

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('AnyPin')
        return helper

    def compute(self, *args, **kwargs):
        # Get the feature names from the input pin
        view = self.view.getData()

        # Print the input data
        #print("Input Data:")
        print(view)

    def createUi(self):
        # Create the main widget for the custom UI
        widget = QtWidgets.QWidget()

        # Create a layout for the widget
        layout = QtWidgets.QVBoxLayout(widget)

        # Create a label to display the pin name
        pin_name_label = QtWidgets.QLabel('view')
        layout.addWidget(pin_name_label)

        # Create a text edit to display the input data
        view_text_edit = QtWidgets.QTextEdit()
        view_text_edit.setReadOnly(True)
        layout.addWidget(view_text_edit)


        # Create a function to update the text edit when the pin data changes
        def update_text_edit(data):
            view_text_edit.setPlainText(str(data))

        self.view.onPinDataChanged.connect(update_text_edit)

        # Set the widget as the custom UI for the node
        self.setWidget(widget)

    @staticmethod
    def category():
        return '2_Data_Visualization'
