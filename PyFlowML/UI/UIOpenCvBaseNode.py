from Qt import QtGui,QtCore
from PyFlow.UI import RESOURCES_DIR
from PyFlow.UI.Canvas.UINodeBase import UINodeBase
from PyFlow.UI.Canvas.NodeActionButton import NodeActionButtonBase
from PyFlow.UI.Canvas.UICommon import *
from Qt.QtWidgets import QLabel
import os
from PyFlow.Packages.PyFlowOpenCv.UI.pc_ImageCanvasWidget import toQImage
import cv2
class ViewImageNodeActionButton(NodeActionButtonBase):
    """docstring for ViewImageNodeActionButton."""
    def __init__(self, svgFilePath, action, uiNode):
        super(ViewImageNodeActionButton, self).__init__(svgFilePath, action, uiNode)
        self.svgIcon.setElementId("Expand")

    def mousePressEvent(self, event):
        super(ViewImageNodeActionButton, self).mousePressEvent(event)
        if not self.parentItem().displayImage:
            self.svgIcon.setElementId("Expand")
        else:
            self.svgIcon.setElementId("Collapse")

class UIOpenCvBaseNode(UINodeBase):
    def __init__(self, raw_node):
        super(UIOpenCvBaseNode, self).__init__(raw_node)
        self.imagePin = self._rawNode.getPinByName("img")
        if not self.imagePin:
            for pin in self._rawNode.outputs.values():
                if pin.dataType == "ImagePin":
                    self.imagePin = pin
                    break
        if  self.imagePin:       
            self.actionViewImage = self._menu.addAction("ViewImage")
            self.actionViewImage.triggered.connect(self.viewImage)
            self.actionViewImage.setData(NodeActionButtonInfo(os.path.dirname(__file__)+"/resources/ojo.svg", ViewImageNodeActionButton))
            self.actionRefreshImage = self._menu.addAction("RefreshCurrentNode")
            self.actionRefreshImage.triggered.connect(self.refreshImage)
            self.actionRefreshImage.setData(NodeActionButtonInfo(os.path.dirname(__file__)+"/resources/reload.svg", NodeActionButtonBase))        
        self.displayImage = False
        self.resizable = True
        self.Imagelabel = QLabel("noImage")
        self.pixmap = QtGui.QPixmap()
        self.addWidget(self.Imagelabel)
        self.Imagelabel.setVisible(False)
        self.updateSize()
        self._rawNode.computed.connect(self.updateImage)

    @property
    def collapsed(self):
        return self._collapsed

    @collapsed.setter
    def collapsed(self, bCollapsed):
        if bCollapsed != self._collapsed:
            self._collapsed = bCollapsed
            self.aboutToCollapse(self._collapsed)
            for i in range(0, self.inputsLayout.count()):
                inp = self.inputsLayout.itemAt(i)
                inp.setVisible(not bCollapsed)
            for o in range(0, self.outputsLayout.count()):
                out = self.outputsLayout.itemAt(o)
                out.setVisible(not bCollapsed)
            for cust in range(0, self.customLayout.count()):
                out = self.customLayout.itemAt(cust)
                out.setVisible(not bCollapsed)
            if not self.displayImage:
                self.Imagelabel.setVisible(False)
            self.updateNodeShape()

    def updateImage(self,*args, **kwargs):
        if self.displayImage and not self.collapsed :
            if self.imagePin:
                img = self.imagePin.getData()
                self.setNumpyArray(img) 

    def refreshImage(self):
        if self.imagePin:
            self._rawNode.processNode()
        if self.displayImage and not self.collapsed :
            if self.imagePin:
                img = self.imagePin.getData()
                self.setNumpyArray(img)
            self.Imagelabel.setVisible(True)
        else:
            self.Imagelabel.setVisible(False)

    def viewImage(self):
        self.displayImage = not self.displayImage
        self.refreshImage()
        self.updateNodeShape()
        self.updateSize()

    def setNumpyArray(self,image):
        if image.__class__.__name__ == "UMat":
            image = cv2.UMat.get(image)        
        image = toQImage(image)
        self.pixmap = QtGui.QPixmap.fromImage(image,QtCore.Qt.ThresholdAlphaDither)
        self.updateSize()

    def paint(self, painter, option, widget):
        self.updateSize()
        super(UIOpenCvBaseNode, self).paint(painter, option, widget)

    def updateSize(self):
        if not self.pixmap.isNull():
            scaledPixmap = self.pixmap.scaledToWidth(self.customLayout.geometry().width() )
            self.Imagelabel.setMaximumWidth(self.customLayout.geometry().width())
            self.Imagelabel.setPixmap(scaledPixmap)

    def updateNodeShape(self):
        super(UIOpenCvBaseNode, self).updateNodeShape()
        self.updateSize()
        self.update()