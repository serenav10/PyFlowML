PACKAGE_NAME = 'PyFlowML'

from collections import OrderedDict
from PyFlow.UI.UIInterfaces import IPackage

# Pins
#from PyFlow.Packages.PyFlowML.Pins.ImagePin import ImagePin

# Function based nodes
from PyFlow.Packages.PyFlowML.FunctionLibraries.MLempty import MLempty
#from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.OpenCvLib import OpenCvLib
#from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.OpticalFlowLib import OpticalFlowLib , LK_optical_flow_Lib, Dense_optical_flow_Lib
#from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.ImageFilteringLib import ImageFilteringLib
#from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.GeometricImageTransformationsLib import GeometricImageTransformationsLib
#from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.ImageBlendingLib import ImageBlendingLib

# Class based nodes
from PyFlow.Packages.PyFlowML.Nodes.Deep_Neural_Network import Deep_Neural_Network
from PyFlow.Packages.PyFlowML.Nodes.Support_Vector_Machine import Support_Vector_Machine
from PyFlow.Packages.PyFlowML.Nodes.K_Nearest_Neighbors import K_Nearest_Neighbors
from PyFlow.Packages.PyFlowML.Nodes.BCWDLoaderNode import BCWDLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.DisplayData import DisplayData
from PyFlow.Packages.PyFlowML.Nodes.SHAP import SHAP
from PyFlow.Packages.PyFlowML.Nodes.LIME import LIME
from PyFlow.Packages.PyFlowML.Nodes.Partial_Dependence_Plot import Partial_Dependence_Plot
from PyFlow.Packages.PyFlowML.Nodes.ICE import ICE
from PyFlow.Packages.PyFlowML.Nodes.Decision_Tree import Decision_Tree
from PyFlow.Packages.PyFlowML.Nodes.Random_Forest import Random_Forest
from PyFlow.Packages.PyFlowML.Nodes.Gradient_Boosting import Gradient_Boosting
from PyFlow.Packages.PyFlowML.Nodes.Multinomial_Naive_Bayes import Multinomial_Naive_Bayes
from PyFlow.Packages.PyFlowML.Nodes.BERT import BERT
from PyFlow.Packages.PyFlowML.Nodes.MNISTLoaderNode import MNISTLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.SMSLoaderNode import SMSLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.MOVLoaderNode import MOVLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.REUTLoaderNode import REUTLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.NEWSLoaderNode import NEWSLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.HDLoaderNode import HDLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.HRLoaderNode import HRLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.TextLoaderNode import TextLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.NumericLoaderNode import NumericLoaderNode
from PyFlow.Packages.PyFlowML.Nodes.RelevantTerms import RelevantTerms
from PyFlow.Packages.PyFlowML.Nodes.Viewer import Viewer
from PyFlow.Packages.PyFlowML.Nodes.Clustering import Clustering
from PyFlow.Packages.PyFlowML.Nodes.Best_Performer import Best_Performer

# Tools
# from PyFlow.Packages.PyFlowML.Tools.ImageViewerTool import ImageViewerTool

# Factories
from PyFlow.Packages.PyFlowML.Factories.PinInputWidgetFactory import getInputWidget
from PyFlow.Packages.PyFlowML.Factories.UINodeFactory import createUINode

_FOO_LIBS = {MLempty.__name__: MLempty(PACKAGE_NAME)
			}


_NODES = {}
_PINS = {}
_TOOLS = OrderedDict()
_PREFS_WIDGETS = OrderedDict()
_EXPORTERS = OrderedDict()


_NODES[Deep_Neural_Network.__name__] = Deep_Neural_Network

_NODES[MNISTLoaderNode.__name__] = MNISTLoaderNode

_NODES[SMSLoaderNode.__name__] = SMSLoaderNode

_NODES[TextLoaderNode.__name__] = TextLoaderNode

_NODES[NumericLoaderNode.__name__] = NumericLoaderNode

_NODES[Support_Vector_Machine.__name__] = Support_Vector_Machine

_NODES[K_Nearest_Neighbors.__name__] = K_Nearest_Neighbors

_NODES[Decision_Tree.__name__] = Decision_Tree

_NODES[Gradient_Boosting.__name__] = Gradient_Boosting

_NODES[Multinomial_Naive_Bayes.__name__] = Multinomial_Naive_Bayes

_NODES[BERT.__name__] = BERT

_NODES[BCWDLoaderNode.__name__] = BCWDLoaderNode

_NODES[MOVLoaderNode.__name__] = MOVLoaderNode

_NODES[REUTLoaderNode.__name__] = REUTLoaderNode

_NODES[NEWSLoaderNode.__name__] = NEWSLoaderNode

_NODES[HDLoaderNode.__name__] = HDLoaderNode

_NODES[HRLoaderNode.__name__] = HRLoaderNode

_NODES[Random_Forest.__name__] = Random_Forest

_NODES[DisplayData.__name__] = DisplayData

_NODES[SHAP.__name__] = SHAP

_NODES[LIME.__name__] = LIME

_NODES[Partial_Dependence_Plot.__name__] = Partial_Dependence_Plot

_NODES[ICE.__name__] = ICE

_NODES[RelevantTerms.__name__] = RelevantTerms

_NODES[Viewer.__name__] = Viewer

_NODES[Clustering.__name__] = Clustering

_NODES[Best_Performer.__name__] = Best_Performer

#_PINS[ImagePin.__name__] = ImagePin

#_TOOLS[ImageViewerTool.__name__] = ImageViewerTool


class PyFlowML(IPackage):
	def __init__(self):
		super(PyFlowML, self).__init__()

	@staticmethod
	def GetExporters():
		return _EXPORTERS

	@staticmethod
	def GetFunctionLibraries():
		return _FOO_LIBS

	@staticmethod
	def GetNodeClasses():
		return _NODES

	@staticmethod
	def GetPinClasses():
		return _PINS

	@staticmethod
	def GetToolClasses():
		return _TOOLS

	#@staticmethod
	#def UIPinsFactory():
	#	return createUIPin

	@staticmethod
	def UINodesFactory():
		return createUINode

	@staticmethod
	def PinsInputWidgetFactory():
		return getInputWidget

