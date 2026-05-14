# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from qgis.gui import QgsMapLayerComboBox
from qgis.core import QgsMapLayerProxyModel  # Import for layer filtering


class Ui_CraterDetectorDialogBase(object):
    def setupUi(self, CraterDetectorDialogBase):
        CraterDetectorDialogBase.setObjectName("CraterDetectorDialogBase")
        CraterDetectorDialogBase.resize(600, 750)  # Standardized height to accommodate elements comfortably
        CraterDetectorDialogBase.setWindowTitle("YOLOLens Analysis Suite")

        self.mainLayout = QtWidgets.QVBoxLayout(CraterDetectorDialogBase)
        self.mainLayout.setSpacing(15)

        # --- HEADER ---
        self.headerFrame = QtWidgets.QFrame()
        self.headerLayout = QtWidgets.QVBoxLayout(self.headerFrame)
        self.labelTitle = QtWidgets.QLabel("YOLOLens")
        fontTitle = QtGui.QFont()
        fontTitle.setPointSize(20)
        fontTitle.setBold(True)
        self.labelTitle.setFont(fontTitle)
        self.labelTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.headerLayout.addWidget(self.labelTitle)

        self.labelDesc = QtWidgets.QLabel("Planetary Crater Detection | ONNX Engine")
        self.labelDesc.setAlignment(QtCore.Qt.AlignCenter)
        self.labelDesc.setStyleSheet("color: #555; font-style: italic;")
        self.headerLayout.addWidget(self.labelDesc)
        self.mainLayout.addWidget(self.headerFrame)

        # --- MODEL SELECTION GROUP ---
        self.groupModel = QtWidgets.QGroupBox("Model Configuration (only for the Moon)")
        self.groupModel.setStyleSheet("font-weight: bold; color: #333;")
        self.modelLayout = QtWidgets.QVBoxLayout(self.groupModel)
        self.modelLayout.setSpacing(10)

        # Container layout for Model 1 Radio Button + Status Bulb
        self.layoutModel1 = QtWidgets.QHBoxLayout()
        self.radioModel1 = QtWidgets.QRadioButton("Model 1 (Standard Optical)")
        self.radioModel1.setChecked(True)
        self.radioModel1.setStyleSheet("font-weight: normal;")
        self.labelModel1Status = QtWidgets.QLabel("●")
        self.labelModel1Status.setFixedWidth(20)
        self.labelModel1Status.setAlignment(QtCore.Qt.AlignCenter)
        self.layoutModel1.addWidget(self.radioModel1)
        self.layoutModel1.addWidget(self.labelModel1Status)
        self.modelLayout.addLayout(self.layoutModel1)

        # Container layout for Model 2 Radio Button + Status Bulb
        self.layoutModel2 = QtWidgets.QHBoxLayout()
        self.radioModel2 = QtWidgets.QRadioButton("Model 2 (Optical, DTM, Hillshade)")
        self.radioModel2.setStyleSheet("font-weight: normal;")
        self.labelModel2Status = QtWidgets.QLabel("●")
        self.labelModel2Status.setFixedWidth(20)
        self.labelModel2Status.setAlignment(QtCore.Qt.AlignCenter)
        self.layoutModel2.addWidget(self.radioModel2)
        self.layoutModel2.addWidget(self.labelModel2Status)
        self.modelLayout.addLayout(self.layoutModel2)

        self.mainLayout.addWidget(self.groupModel)

        # --- INPUT & CONFIGURATION GROUP ---
        self.groupLayers = QtWidgets.QGroupBox("Inputs & Configuration")
        self.groupLayers.setStyleSheet("font-weight: bold; color: #333;")
        self.layersLayout = QtWidgets.QGridLayout(self.groupLayers)
        self.layersLayout.setSpacing(10)

        # Optical Layer
        self.labelOptical = QtWidgets.QLabel("Optical Orthomosaic:")
        self.labelOptical.setStyleSheet("font-weight: normal;")
        self.layersLayout.addWidget(self.labelOptical, 0, 0)
        self.mMapLayerComboBox_Optical = QgsMapLayerComboBox()
        self.mMapLayerComboBox_Optical.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layersLayout.addWidget(self.mMapLayerComboBox_Optical, 0, 1)

        # DTM Layer (Disabled by default, enabled only when model 2 is chosen)
        self.labelDTM = QtWidgets.QLabel("Input DTM:")
        self.labelDTM.setStyleSheet("font-weight: normal;")
        self.labelDTM.setEnabled(False)
        self.layersLayout.addWidget(self.labelDTM, 1, 0)
        self.mMapLayerComboBox_DTM = QgsMapLayerComboBox()
        self.mMapLayerComboBox_DTM.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.mMapLayerComboBox_DTM.setEnabled(False)
        self.layersLayout.addWidget(self.mMapLayerComboBox_DTM, 1, 1)

        # Confidence Threshold - Cleanly integrated into the layout grid below inputs
        self.labelConf = QtWidgets.QLabel("Confidence Threshold:")
        self.labelConf.setStyleSheet("font-weight: normal;")
        self.layersLayout.addWidget(self.labelConf, 2, 0)

        self.doubleSpinBoxConf = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBoxConf.setRange(0.0, 1.0)
        self.doubleSpinBoxConf.setSingleStep(0.05)
        self.doubleSpinBoxConf.setValue(0.25)
        self.layersLayout.addWidget(self.doubleSpinBoxConf, 2, 1)

        self.mainLayout.addWidget(self.groupLayers)

        # --- EXECUTION METRICS GROUP ---
        self.groupMetrics = QtWidgets.QGroupBox("Execution Metrics")
        self.groupMetrics.setStyleSheet("font-weight: bold; color: #333;")
        self.metricsLayout = QtWidgets.QGridLayout(self.groupMetrics)
        self.metricsLayout.setSpacing(10)

        self.metricsLayout.addWidget(QtWidgets.QLabel("Available Devices:"), 0, 0)
        self.labelDevice = QtWidgets.QLabel("Detecting...")
        self.labelDevice.setStyleSheet("font-family: 'Courier New'; font-weight: bold; color: #27ae60;")
        self.metricsLayout.addWidget(self.labelDevice, 0, 1)

        self.metricsLayout.addWidget(QtWidgets.QLabel("Status Code:"), 1, 0)
        self.labelStatus = QtWidgets.QLabel("Status: Ready")
        self.labelStatus.setStyleSheet("font-weight: bold; color: #2980b9;")
        self.metricsLayout.addWidget(self.labelStatus, 1, 1)

        self.metricsLayout.addWidget(QtWidgets.QLabel("Execution Latency:"), 2, 0)
        self.labelTime = QtWidgets.QLabel("0.00 ms")
        self.labelTime.setStyleSheet("font-family: 'Courier New'; font-weight: bold; color: #c0392b;")
        self.metricsLayout.addWidget(self.labelTime, 2, 1)

        self.loaderLabel = QtWidgets.QLabel()
        self.loaderLabel.setFixedSize(160, 160)
        self.loaderLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.metricsLayout.addWidget(self.loaderLabel, 1, 2, 2, 1)
        self.loaderLabel.hide()

        self.mainLayout.addWidget(self.groupMetrics)

        # --- PROGRESS BAR ---
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.mainLayout.addWidget(self.progressBar)

        # --- ACTION BUTTONS ---
        self.pushButton = QtWidgets.QPushButton("Execute Analysis")
        self.pushButton.setMinimumHeight(45)
        self.pushButton.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; border-radius: 5px;")
        self.mainLayout.addWidget(self.pushButton)

        QtCore.QMetaObject.connectSlotsByName(CraterDetectorDialogBase)