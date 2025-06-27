import logging
import os
from typing import Annotated, Optional

import vtk

import ctk 

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLTransformNode
from slicer import vtkMRMLVectorVolumeNode
import vtk.util.numpy_support
import qt

#
# BrainShiftModule
#
def setCrosshairColor(colorRGB):
    layoutManager = slicer.app.layoutManager()
    sliceViewNames = slicer.util.getSliceViewNames()  # ['Red', 'Yellow', 'Green']
    
    for viewName in sliceViewNames:
        sliceWidget = layoutManager.sliceWidget(viewName)
        dm = sliceWidget.sliceView().displayableManagerByClassName("vtkMRMLCrosshairDisplayableManager")
        if dm:
            # Update color: (R,G,B) in range 0.0-1.0
            dm.SetCrosshairColor(colorRGB)


class BrainShiftModule(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("BrainShiftModule")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = [] 
        self.parent.contributors = ["Elise Donszelmann-Lund (McGill), Isabel Frolick (McGill), Étienne Léger"]  
        self.parent.helpText = _("""
            Visualize Brain Shift (mm) per voxel
            See more information in <a href="https://github.com/organization/projectname#BrainShiftModule">module documentation</a>.
            """)
        self.parent.acknowledgementText = _(""" """)

#
# BrainShiftModuleParameterNode
#


@parameterNodeWrapper
class BrainShiftModuleParameterNode:
    """
    The parameters needed by module.


    referenceVolume - The pre-operative MRI volume
    transformationNode - The transformation applied (MRI to iUS)).
    displacementField - volume node to store 3D vector field
    displacementMagnitudeVolume - Output volume storing per-voxel displacement (magnitude) in mm.
    """

    referenceVolume: vtkMRMLScalarVolumeNode
    transformNode: vtkMRMLTransformNode
    displacementMagnitudeVolume: vtkMRMLScalarVolumeNode
    backgroundVolume: vtkMRMLScalarVolumeNode



#
# BrainShiftModuleWidget
#


class BrainShiftModuleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/BrainShiftModule.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # set the scene for each individual node widget
        self.ui.referenceVolume.setMRMLScene(slicer.mrmlScene)
        self.ui.transformNode.setMRMLScene(slicer.mrmlScene)
        self.ui.displacementMagnitudeVolume.setMRMLScene(slicer.mrmlScene)
        self.ui.backgroundVolume.setMRMLScene(slicer.mrmlScene)
        #self.ui.ConvertTagFCSVNode.setMRMLScene(slicer.mrmlScene)

        self.ui.backgroundVolume.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ui.backgroundVolume.addEnabled = False 
        self.ui.backgroundVolume.removeEnabled = False

        self.ui.referenceVolume.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ui.referenceVolume.addEnabled = False
        self.ui.referenceVolume.removeEnabled = False

        self.ui.displacementMagnitudeVolume.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ui.displacementMagnitudeVolume.addEnabled = True  # can create new output volume
        self.ui.displacementMagnitudeVolume.removeEnabled = True

        self.ui.transformNode.nodeTypes = ["vtkMRMLTransformNode"]
        self.ui.transformNode.addEnabled = False
        self.ui.transformNode.removeEnabled = False 

        #self.ui.ConvertTagFCSVNode.nodeTypes = ["vtkMRMLTransformNode"]
        self.ui.transformNode.addEnabled = False
        self.ui.transformNode.removeEnabled = False 

        self.ui.MRMLReplacementVolume.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ui.MRMLReplacementVolume.setMRMLScene(slicer.mrmlScene)

        # connect
        self.ui.loadDisplacementVolumeButton.connect("clicked(bool)", self.onLoadDisplacementVolume)

        # color selector
        self.ui.colorMapSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.colorMapSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.colorMapSelector.nodeTypes = [
            "vtkMRMLColorTableNode",
            "vtkMRMLProceduralColorNode",
            "vtkMRMLPETColorNode"
        ]

        # mouse displayer
        self.crosshairNode = slicer.util.getNode("Crosshair")

        self.labelMarkupNode = slicer.util.getModule("Data").mrmlScene().GetFirstNodeByName("BrainShiftModule_MouseValueLabel")
        if(  not self.labelMarkupNode ):
            self.labelMarkupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "BrainShiftModule_MouseValueLabel")
            self.labelMarkupNode.AddControlPoint(0, 0, 0)
            self.labelMarkupNode.SetLocked(True)
            self.labelMarkupNode.SetMarkupLabelFormat("{label}")
            self.labelMarkupNode.GetDisplayNode().SetVisibility2D(False)
            self.labelMarkupNode.GetDisplayNode().SetVisibility3D(False)
            self.labelMarkupNode.SetNthControlPointLabel(0, "")
            self.labelMarkupNode.GetDisplayNode().SetColor([0.0, 0.0, 0.0])  # [0.0,0.0,0.0]       # Fiducial marker color
            self.labelMarkupNode.GetDisplayNode().SetSelectedColor([0.0, 0.0, 0.0])  # [0.0, 0.0, 0.0]    # Color when selected
            self.labelMarkupNode.GetDisplayNode().GetTextProperty().SetColor(0.0, 0.0, 0.0)  # 0,0,0 # Label **text** color (this is key!)

        self.crosshairObserverTag = None

        self.ui.enableHoverDisplayCheckbox.setChecked(False)  # start disabled
        self.ui.enableHoverDisplayCheckbox.connect("toggled(bool)", self.onToggleHoverDisplay)


        # connect backgroundVolume
        self.ui.backgroundVolume.setProperty("SlicerParameterName", "backgroundVolume")

        # connect US Border display checkbox
        self.ui.enableUsBorderDisplay.toggled.connect(self.onToggleUsDisplay)
        
        self.ui.displayLandmarks.toggled.connect(self.onToggleLandmarkDisplay)

        
        # connect threshold slider
        self.ui.thresholdSlider.connect("valuesChanged(double,double)", self.onThresholdSliderChanged)
        # set spin box max and mins
        self.ui.thresholdMinSpinBox.connect("valueChanged(double)", self.onMinSpinBoxChanged)
        self.ui.thresholdMaxSpinBox.connect("valueChanged(double)", self.onMaxSpinBoxChanged)        
        # self.TagFilePathObj = ctk.ctkPathLineEdit()
        # self.TagFilePathObj.filters = ctk.ctkPathLineEdit.Files
        # self.TagFilePathObj.nameFilters = ["Tag files (*.tag)"]
        # self.TagFilePathObj.setToolTip("Select a .tag file to convert")
        # #self.layout.addWidget(self.TagFilePathObj)

        # selectedPath = self.TagFilePathObj.currentPath
        # #formLayout.addRow("Input .tag file:", self.TagFilePathObj)
        # print("Selected file:", selectedPath)

        # button to create fcsv from tag file
        self.ConvertTagFCSVButton = qt.QPushButton("Load Tag File")
        self.ConvertTagFCSVButton.toolTip = "Load a .tag file and create fiducial nodes"
        #self.layout.addWidget(self.ConvertTagFCSVButton)
        #self.ConvertTagFCSVButton.clicked.connect(self.onConvertTagFCSVButtonClicked)
        self.ui.ConvertTagFCSVButton.connect("clicked(bool)", self.onConvertTagFCSVButtonClicked)


        #Visualize the landmarks as desired (multi-slect)
        self.LandmarkSelectorComboBox = ctk.ctkCheckableComboBox()
        self.LandmarkSelectorComboBox.setToolTip("Select fiducial landmark nodes to display")
        #self.layout.addWidget(self.LandmarkSelectorComboBox)
        
        # Populate list manually
        self.updateLandmarkSelectorComboBox()

        # Connect change signal
        self.LandmarkSelectorComboBox.connect('checkedIndexesChanged()', self.onLandmarkSelectionChanged)

        self.LoadExpertLabelsButton = qt.QPushButton("Load Tag File")
        self.LoadExpertLabelsButton.toolTip = "Visualize landmarks"
        #self.layout.addWidget(self.LoadExpertLabelsButton)
        self.ui.LoadExpertLabelsButton.connect('clicked(bool)', self.onLoadExpertLabelsClicked)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = BrainShiftModuleLogic()

        # Connections
        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
       
    
        #Add observer to the node event
        #slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, lambda caller, event: self.updateLandmarkSelectorComboBox())
        #self.addObserver(slicer.mrmlScene, slicer.vtkMRMLScene.NodeAddedEvent, self.updateLandmarkSelectorComboBox() )
        
        self.addObserver(slicer.mrmlScene, slicer.vtkMRMLScene.NodeAddedEvent, self.onNodeAdded)
        #self.addObserver(slicer.mrmlScene, slicer.vtkMRMLScene.EndBatchProcessEvent, self.onSceneUpdated)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # allow for user to adjust opacity
        self.ui.opacitySlider.valueChanged.connect(self.onOpacityChanged)


    def onOpacityChanged(self, value) -> None:
        normalizedValue = value/100
        slicer.util.setSliceViewerLayers(foregroundOpacity=normalizedValue)

    def onToggleUsDisplay(self) -> None:
        usVolume = self.ui.referenceVolume.currentNode()
        state = self.ui.enableUsBorderDisplay.checkState() 
    
        self.logic.showNonZeroWireframe(foregroundVolume=usVolume, state=state)

    def onToggleLandmarkDisplay(self) -> None:
        #usVolume = self.ui.referenceVolume.currentNode()
        state = self.ui.enableUsBorderDisplay.checkState()
    
        #
        # self.logic.showNonZeroWireframe(foregroundVolume=usVolume, state=state)


    def onConvertTagFCSVButtonClicked(self):
        filePath = qt.QFileDialog.getOpenFileName(
            None, "Open Tag File", "", "Tag files (*.tag)"
        )
        print("Selected file:", filePath)
        if filePath:
            success = self.logic.loadTagFile(filePath)
            if not success:
                slicer.util.errorDisplay(f"Failed to load tag file: {filePath}")
            else:
                logging.info(f"Loaded tag file: {filePath}")
        

    def onLoadExpertLabelsClicked(self):
        '''
        Select which landmarks to visualize from Data module
        '''
        selectedNames = []
        comboBox = self.ui.LandmarkSelectorComboBox  # or self.LandmarkSelectorComboBox

        for i in range(comboBox.count):
            index = comboBox.model().index(i, 0)
            if comboBox.checkState(index) == qt.Qt.Checked:
                selectedNames.append(comboBox.itemText(i))

                fiducialNode = slicer.util.getNode(comboBox.itemText(i))

                # Ensure it is a fiducial node
                if fiducialNode.IsA("vtkMRMLMarkupsFiducialNode"):
                    #if self.ui.enableLandmarkDisplay.checkState():
                        # Turn on visibility in 3D view
                    fiducialNode.GetDisplayNode().SetVisibility(True)
                    fiducialNode.GetDisplayNode().SetVisibility2D(True)

                    fiducialNode.GetDisplayNode().SetGlyphScale(3.0) #Marker
                    fiducialNode.GetDisplayNode().SetTextScale(3.0) #Label

                    fiducialNode.GetDisplayNode().SetActiveColor([1.0, 0.2, 0.5])   # Pink when active
                    #fiducialNode.GetDisplayNode().SetColor(0.0, 0.0, 0.0)           # Pink when not active
                    fiducialNode.GetDisplayNode().SetSelectedColor(0.0, 0.0, 0.0)   # Pink when selected
                    #fiducialNode.GetDisplayNode().SetGlyphType(0)
                    fiducialNode.GetDisplayNode().SetGlyphTypeFromString("Circle2D")  # Hide glyph icon

                    fiducialNode.GetDisplayNode().SetSelected(True)
                    fiducialNode.GetDisplayNode().SetHandlesInteractive(False)
                else:
                    fiducialNode.GetDisplayNode().SetVisibility(False)
                    fiducialNode.GetDisplayNode().SetVisibility2D(False)


    
    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()
        for interactor, tag in getattr(self, "sliceObservers", []):
            interactor.RemoveObserver(tag)
            self.sliceObservers = []
    

    def onSceneUpdated(self, caller, event):
        self.updateLandmarkSelectorComboBox()
    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())
    
  
    def onNodeAdded(self, caller, event, callData):
        newNode = callData
        print("In onNodeAdded")
        if isinstance(newNode, slicer.vtkMRMLMarkupsFiducialNode):
            print(f"New fiducial node added: {newNode.GetName()}")
            self.updateLandmarkSelectorComboBox()
                
    def updateLandmarkSelectorComboBox(self):
        '''
        Tracks which files to add to the selection box for the available landmarks
        '''
        
        self.LandmarkSelectorComboBox.clear()
        print("Update... ")
        fiducialNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        print("fiducial Nodes available", len(fiducialNodes))

        for node in fiducialNodes:
            self.ui.LandmarkSelectorComboBox.addItem(node.GetName())


    def onLandmarkSelectionChanged(self):
        # Get all fiducial nodes
        allFiducials = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")

        # Get selected names from the combo box
        selectedNames = []
        for i in range(self.LandmarkSelectorComboBox.count):
            if self.LandmarkSelectorComboBox.checkState(i) == qt.Qt.Checked:
                selectedNames.append(self.LandmarkSelectorComboBox.itemText(i))

        # Show only selected ones
        for node in allFiducials:
            displayNode = node.GetDisplayNode()
            if not displayNode:
                continue
            if node.GetName() in selectedNames:
                #if self.ui.enableLandmarkDisplay.checkState(): #onToggleLandmarkDisplay

                displayNode.SetUsePointColors(False)         # Use global color, not per-point
                displayNode.SetVisibility(True)
                displayNode.SetVisibility2D(True)
                displayNode.SetTextScale(1.0)
                
                displayNode.SetActiveColor([1.0, 0.0, 1.0])   # Pink when active
                displayNode.SetColor(1.0, 0.0, 1.0)           # Pink when not active
                displayNode.SetSelectedColor(1.0, 0.0, 1.0)   # Pink when selected
                displayNode.SetUseSelectedColor(True)       
                
                displayNode.SetGlyphScale(2.0)
                displayNode.SetHandlesInteractive(False)
            else:
                displayNode.SetVisibility(False)
                displayNode.SetVisibility2D(False)


    def setParameterNode(self, inputParameterNode: Optional[BrainShiftModuleParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        
        # make sure there's a reference MRI and transformation
        if (
            self._parameterNode
            and self._parameterNode.referenceVolume
            and self._parameterNode.transformNode
            and self._parameterNode.backgroundVolume
        ):
            self.ui.applyButton.toolTip = _("Compute voxel-wise displacement magnitude")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select reference volume and transform")
            self.ui.applyButton.enabled = False


    def onApplyButton(self) -> None:
        """Run processing when user clicks 'Apply' button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute voxel-wise displacement."), waitCursor=True):

            logging.info(f"Reference Volume: {self._parameterNode.referenceVolume}")
            logging.info(f"Transform Node: {self._parameterNode.transformNode}")
            logging.info(f"Displacement Volume: {self._parameterNode.displacementMagnitudeVolume}")


            # Create displacement field (vector volume)
            self.logic.computeDisplacementMagnitude(
                referenceVolume=self._parameterNode.referenceVolume,
                transformNode=self._parameterNode.transformNode,
                outputVolume=self._parameterNode.displacementMagnitudeVolume
            )
            slicer.util.setSliceViewerLayers(
                # background=self._parameterNode.referenceVolume,
            
                background=self._parameterNode.backgroundVolume,
                foreground=self._parameterNode.displacementMagnitudeVolume
                                
            )
            
            # change to color thats selected
            colorNode = self.ui.colorMapSelector.currentNode()
            if colorNode and self._parameterNode.displacementMagnitudeVolume:
                displayNode = self._parameterNode.displacementMagnitudeVolume.GetDisplayNode()
                if displayNode:
                    displayNode.SetAndObserveColorNodeID(colorNode.GetID())


    def onLoadDisplacementVolume(self) -> None:

        #selectedVolume = self.ui.existingDisplacementVolumeSelector.currentNode()
        selectedVolume = self.ui.MRMLReplacementVolume.currentNode()
        usVolume = self.ui.referenceVolume.currentNode()

        # referenceVolume = self._parameterNode.referenceVolume
        backgroundVolume = self._parameterNode.backgroundVolume
        
        state = self.ui.enableUsBorderDisplay.checkState()
        self.logic.showNonZeroWireframe(foregroundVolume=usVolume, state=state, reload=True)
        # visualize it
        slicer.util.setSliceViewerLayers(
            background=backgroundVolume,
            foreground=selectedVolume
        )
        
        self.onLoadExpertLabelsClicked()

        # change to selected color
        colorNode = self.ui.colorMapSelector.currentNode()
        if colorNode:
            displayNode = selectedVolume.GetDisplayNode()
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())
            displayNode.Modified()

        normalizedValue = self.ui.opacitySlider.value / 100
        slicer.util.setSliceViewerLayers(foregroundOpacity=normalizedValue)

        # set max and min of threshold slider
        imageData = selectedVolume.GetImageData()
        if imageData:
            scalarRange = imageData.GetScalarRange()
            minScalar, maxScalar = 0, scalarRange[1]

            # set threshold slider limits based on max and min displacement values
            #self.ui.thresholdSlider.setMinimum(minScalar)
            #self.ui.thresholdSlider.setMaximum(maxScalar)
            self.ui.thresholdSlider.minimum = minScalar
            self.ui.thresholdSlider.maximum = maxScalar
            self.ui.thresholdSlider.setMinimumValue(minScalar)
            self.ui.thresholdSlider.setMaximumValue(maxScalar)
            self.ui.thresholdSlider.setValues(0.5, maxScalar)

            self.ui.thresholdMinSpinBox.setMinimum(minScalar)
            self.ui.thresholdMinSpinBox.setMaximum(maxScalar)
            self.ui.thresholdMinSpinBox.setValue(0.5)

            self.ui.thresholdMaxSpinBox.setMinimum(minScalar)
            self.ui.thresholdMaxSpinBox.setMaximum(maxScalar)
            self.ui.thresholdMaxSpinBox.setValue(maxScalar)
        




    def onMouseMoved(self, observer, eventid):
        # if markup node doesn't exist do nothing
        if not self.labelMarkupNode.GetDisplayNode().GetVisibility2D():
            return

        ras = [0.0, 0.0, 0.0]
        self.crosshairNode.GetCursorPositionRAS(ras)
    
        # move label to current RAS position
        self.labelMarkupNode.SetNthControlPointPosition(0, ras)

        # sample displacement volume at that RAS location
        #displacementVolume = self.ui.existingDisplacementVolumeSelector.currentNode()
        displacementVolume = self.ui.MRMLReplacementVolume.currentNode()

        if not displacementVolume:
            self.labelMarkupNode.SetNthControlPointLabel(0, "No volume")
            return

        # convert RAS to IJK
        rasToIjk = vtk.vtkMatrix4x4()
        displacementVolume.GetRASToIJKMatrix(rasToIjk)
        ijk = [0.0, 0.0, 0.0, 1.0]
        ras_hom = list(ras) + [1.0]
        rasToIjk.MultiplyPoint(ras_hom, ijk)
        ijk = [int(round(i)) for i in ijk[:3]]

        dims = displacementVolume.GetImageData().GetDimensions()
        if any(i < 0 or i >= d for i, d in zip(ijk, dims)):
            self.labelMarkupNode.SetNthControlPointLabel(0, "Out of bounds")
            return

        value = displacementVolume.GetImageData().GetScalarComponentAsDouble(*ijk, 0)
        label = f"{value:.3f} mm"
        self.labelMarkupNode.SetNthControlPointLabel(0, label)


    def onToggleHoverDisplay(self, enabled: bool) -> None:

        if enabled:
            self.labelMarkupNode.GetDisplayNode().SetVisibility2D(True)
            # add observer if not already observing
            if self.crosshairObserverTag is None:
                self.crosshairObserverTag = self.crosshairNode.AddObserver(
                    slicer.vtkMRMLCrosshairNode.CursorPositionModifiedEvent,
                    self.onMouseMoved
                )
        else:
            self.labelMarkupNode.GetDisplayNode().SetVisibility2D(False)
            # remove observer if it exists
            if self.crosshairObserverTag is not None:
                self.crosshairNode.RemoveObserver(self.crosshairObserverTag)
                self.crosshairObserverTag = None


    def onThresholdSliderChanged(self, minValue, maxValue):

        volumeNode = self.ui.MRMLReplacementVolume.currentNode()
        if not volumeNode:
            logging.warning("No displacement magnitude volume available for thresholding.")
            return
        
        # dynamically set min and max value
        displayNode = volumeNode.GetDisplayNode()

        displayNode.AutoWindowLevelOff()
        displayNode.SetThreshold(minValue, maxValue)
        displayNode.SetApplyThreshold(True)
        displayNode.Modified()
        logging.info(f"Threshold applied: min = {minValue}, max = {maxValue}")

        self.ui.thresholdMinSpinBox.blockSignals(True)
        self.ui.thresholdMaxSpinBox.blockSignals(True)
        self.ui.thresholdMinSpinBox.setValue(minValue)
        self.ui.thresholdMaxSpinBox.setValue(maxValue)
        self.ui.thresholdMinSpinBox.blockSignals(False)
        self.ui.thresholdMaxSpinBox.blockSignals(False)


    def onMinSpinBoxChanged(self, value):
        currentMax = self.ui.thresholdMaxSpinBox.value
        self.ui.thresholdSlider.setValues(value, currentMax)

    def onMaxSpinBoxChanged(self, value):
        currentMin = self.ui.thresholdMinSpinBox.value
        self.ui.thresholdSlider.setValues(currentMin, value)


# BrainShiftModuleLogic
#


class BrainShiftModuleLogic(ScriptedLoadableModuleLogic):
    """Logic for computing voxel-wise displacement from transformation field"""

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return BrainShiftModuleParameterNode(super().getParameterNode())
    
    def countUniqueValues(self, volumeNode: vtkMRMLScalarVolumeNode):

        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        imageData = volumeNode.GetImageData()
        if imageData is None:
            logging.warning("Volume has no image data.")
            return None

        vtk_array = imageData.GetPointData().GetScalars()
        np_array = vtk_to_numpy(vtk_array)

        unique_values = np.unique(np_array)
        logging.info(f"Number of unique values in displacement magnitude volume: {len(unique_values)}")
        return len(unique_values), unique_values

    def loadTagFile(self, filepath):
        print(f"Reading tag file: {filepath}")

        points1, points2 = self.read_tag_file(filepath)
        if points1 is None or points2 is None or len(points1) == 0 or len(points2) == 0:
            logging.error("No valid points found in tag file.")
            return False

        # create fiducial nodes in Slicer scene
        fiducialNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Source_Landmarks") #the set of landmarks from the first volume registered
        fiducialNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Target_Landmarks") #the set of landmarks from the second volume registered

        # add points from tag file to the fiducial nodes
        for pt in points1:
            fiducialNode1.AddControlPoint(pt)
        for pt in points2:
            fiducialNode2.AddControlPoint(pt)

        displayNode1 = fiducialNode1.GetDisplayNode()
        if displayNode1:
            displayNode1.SetVisibility(False)            # Hide in 3D
            displayNode1.SetVisibility2D(False)          # Hide in 2D slice views
            displayNode1.SetSelectedColor(0.5, 0.5, 0.5) # Optional: make it less prominent when turned on
            displayNode1.SetTextScale(5.0)               # Hide label text
            #displayNode1.SetGlyphTypeFromString("None")  # Hide glyph icon
            displayNode1.SetHandlesInteractive(False)    # Disable user interaction
            #displayNode1.SetOpacity(1.0)
            #displayNode1.SetGlyphScale(5.0)  
        print("Number of points1:", fiducialNode1.GetNumberOfControlPoints())

        displayNode2 = fiducialNode2.GetDisplayNode()
        if displayNode2:
            displayNode2.SetVisibility(False)            # Hide in 3D
            displayNode2.SetVisibility2D(False)          # Hide in 2D slice views
            displayNode2.SetSelectedColor(0.5, 0.5, 0.5) # Optional: make it less prominent when turned on
            displayNode2.SetTextScale(5.0)               # Hide label text
            displayNode2.SetOpacity(1.0)
            displayNode2.SetGlyphScale(5.0)  
            #displayNode2.SetGlyphTypeFromString("None")  # Hide glyph icon
            displayNode2.SetHandlesInteractive(False)    # Disable user interaction
        
       
        else:
            slicer.util.errorDisplay("Failed to load landmark file.")
        qt.QMessageBox.information(slicer.util.mainWindow(), "Success", "Success! \nLandmark files created and available in Data.")
        print("Number of points2:", fiducialNode1.GetNumberOfControlPoints())

        logging.info("Landmarks loaded and hidden.")
        
        logging.info(f"Created {len(points1)} landmarks in each set.")
        return True

    def read_tag_file(self, filepath):
        import numpy as np
        import re

        """Parse the tag file robustly and return two numpy arrays of points."""
        source_points = []
        target_points = []
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('%') or not line:
                        continue
                    try:
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        if len(values) >= 6:
                            source_points.append(values[0:3])
                            target_points.append(values[3:6])
                    except ValueError:
                        # skip lines that cannot be parsed into floats
                        continue
        except Exception as e:
            logging.error(f"Failed to read tag file: {e}")
            return None, None
        return np.array(source_points), np.array(target_points)

    def computeDisplacementMagnitude(self,
                                 referenceVolume: vtkMRMLScalarVolumeNode,
                                 transformNode: vtkMRMLTransformNode,
                                 outputVolume: vtkMRMLScalarVolumeNode) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        """


        if not referenceVolume or not transformNode or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        import numpy as np 
        import vtk
        from vtk.util import numpy_support
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


        startTime = time.time()
        logging.info("Displacement computation started")

        # fill this in

        imageData = referenceVolume.GetImageData()
        logging.info(f"Extenent check: {imageData.GetExtent()}")

        if imageData is None:
            logging.error("Reference volume has no image data")
            return
        dims = imageData.GetDimensions()
        logging.info(f"Reference volume dims: {dims}")

        # ui check
        logging.info(f"referenceVolume name: {referenceVolume.GetName()}")


        spacing = referenceVolume.GetSpacing()
        origin = referenceVolume.GetOrigin()
        logging.info(f"Reference origin: {origin}")

        # Set up transform
        transformToWorld = vtk.vtkGeneralTransform()
        transformNode.GetTransformToWorld(transformToWorld)

        # prepare output image
        magnitudeImage = vtk.vtkImageData()
        magnitudeImage.SetDimensions(dims)
        magnitudeImage.AllocateScalars(vtk.VTK_FLOAT, 1)
        magnitudeImage.SetExtent(imageData.GetExtent())


        # iterate over each voxel in reference image
        for z in range(dims[2]):
            for y in range(dims[1]):
                for x in range(dims[0]):
                    # Voxel coordinate in RAS
                    ras = [origin[0] + x * spacing[0],
                        origin[1] + y * spacing[1],
                        origin[2] + z * spacing[2]]
                    
                    transformedPoint = transformToWorld.TransformPoint(ras)
                    displacement = np.array(transformedPoint) - np.array(ras)
                    magnitude = np.linalg.norm(displacement)
                    magnitudeImage.SetScalarComponentFromFloat(x, y, z, 0, magnitude)

        outputVolume.SetAndObserveImageData(magnitudeImage)
        outputVolume.CopyOrientation(referenceVolume)
        outputVolume.SetSpacing(referenceVolume.GetSpacing())
        outputVolume.SetOrigin(referenceVolume.GetOrigin())
        outputVolume.Modified()


        # until here

        num_unique, unique_vals = self.countUniqueValues(outputVolume)
        print(f"Unique values count: {num_unique}")
        print(f"Unique values count: {num_unique}")

        # enhance display with color map
        if not outputVolume.GetDisplayNode():
            slicer.modules.volumes.logic().CreateDefaultDisplayNodes(outputVolume)
        displayNode = outputVolume.GetDisplayNode()
        displayNode.AutoWindowLevelOff()
        displayNode.SetWindow(10.0)
        displayNode.SetLevel(5.0)
      

        displayNode.SetThreshold(0.05, 10.0)
        displayNode.SetApplyThreshold(True)

        colorNode = slicer.util.getNode("Inferno")
        if colorNode:
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())

        

        logging.info(f"Displacement computation completed in {time.time() - startTime:.2f} s")

    
    def showNonZeroWireframe(self, foregroundVolume, state, reload=False, modelName="NonZeroWireframe"):
        """
        Extracts the non-zero region of a volume and displays its surface wireframe
        as a non-destructive 3D overlay using a vtkModelNode.
        """
        import slicer
        import vtk
        import numpy as np
        import SimpleITK as sitk
        import sitkUtils
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        currentModelNode = slicer.mrmlScene.GetFirstNodeByName(modelName)
        if currentModelNode:
            currentDisplayNode = currentModelNode.GetDisplayNode()
            if not reload:
                if not state:
                    currentDisplayNode.SetVisibility2D(False)
                    return
                currentDisplayNode.SetVisibility2D(True)
                return

            slicer.mrmlScene.RemoveNode(currentDisplayNode)
            slicer.mrmlScene.RemoveNode(currentModelNode)

        print("Starting showNonZeroWireframe...")
        # Step 1: Convert foreground image to binary mask
        print("Step 1: Pulling volume from Slicer for node: %s", foregroundVolume.GetName())
        image_sitk = sitkUtils.PullVolumeFromSlicer(foregroundVolume)
        arr = sitk.GetArrayFromImage(image_sitk)
        nonzero_voxels = np.count_nonzero(arr)
        print("Step 1b: Number of non-zero voxels: %d", nonzero_voxels)

        if nonzero_voxels == 0:
            logger.warning("WARNING: No non-zero voxels found in the image. Aborting wireframe display.")
            return

        mask_arr = (arr != 0).astype(np.uint8)
        mask_sitk = sitk.GetImageFromArray(mask_arr)
        mask_sitk.CopyInformation(image_sitk)
        print("Step 1b: Converted to binary mask.")

        # Push to Slicer as labelmap
        labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempBinaryMask")
        sitkUtils.PushVolumeToSlicer(mask_sitk, targetNode=labelNode)
        print("Step 1 done: Binary mask pushed to Slicer label node: %s", labelNode.GetName())

        # Step 2: Extract surface using marching cubes
        imageData = labelNode.GetImageData()
        if imageData is None:
            print("WARNING: Image data is None. Check label node data.")
            return

        print("Step 2: Starting marching cubes...")
        marching = vtk.vtkDiscreteMarchingCubes()
        marching.SetInputData(imageData)
        marching.SetValue(0, 1)
        marching.Update()
        surface = marching.GetOutput()
        if surface is None or surface.GetNumberOfPoints() == 0:
            print("WARNING: No surface generated by marching cubes.")
            return

        # Step 3: Extract wireframe edges from the surface mesh
        print("Step 3: Extracting edges from surface...")
        edges = vtk.vtkExtractEdges()
        edges.SetInputConnection(marching.GetOutputPort())
        edges.Update()

        # Step 4: Create and show model node with only edges
        print("Step 4: Preparing model node '%s'...", modelName)
        newModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", modelName)
        print("STep 4 done: Created new model node.")
        newModelNode.SetAndObservePolyData(edges.GetOutput())

        # Match spatial transform
        print("Step 5: Applying spatial transform (IJK to RAS)...")
        transform = vtk.vtkTransform()
        ijkToRAS = vtk.vtkMatrix4x4()
        labelNode.GetIJKToRASMatrix(ijkToRAS)
        transform.SetMatrix(ijkToRAS)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(edges.GetOutput())
        transformFilter.Update()
        newModelNode.SetAndObservePolyData(transformFilter.GetOutput())

        # Step 6: Set wireframe-only display
        print("Step 6: Setting display properties for wireframe...")
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        newModelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        print("Done: Created and linked display node.")

        displayNode.SetRepresentation(0)  # Wireframe
        displayNode.SetColor(0, 0, 0)     # Green
        displayNode.SetEdgeVisibility(True)
        displayNode.SetPointSize(3.0)
        displayNode.SetSliceIntersectionThickness(2)

        displayNode.SetVisibility3D(False)
        displayNode.SetVisibility2D(True)

        # Optional: remove temp label node
        slicer.mrmlScene.RemoveNode(labelNode)
        print("Temporary label node removed. Done!")

        return newModelNode
