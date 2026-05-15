# -*- coding: utf-8 -*-
import os
from qgis.PyQt import QtWidgets
from .crater_detector_dialog_base import Ui_CraterDetectorDialogBase


class CraterDetectorDialog(QtWidgets.QDialog, Ui_CraterDetectorDialogBase):
    def __init__(self, parent=None):
        super(CraterDetectorDialog, self).__init__(parent)
        self.setupUi(self)
        self.curr_dir = os.path.dirname(__file__)
        self.loaderLabel.setScaledContents(True)
        self.radioModel1.toggled.connect(self.update_ui_state)
        self.radioModel2.toggled.connect(self.update_ui_state)
        self.check_models_exist()
        self.update_ui_state()

    def update_ui_state(self):
        is_multichannel = self.radioModel2.isChecked()
        self.labelDTM.setEnabled(is_multichannel)
        self.mMapLayerComboBox_DTM.setEnabled(is_multichannel)

    def check_models_exist(self):
        models_dir = os.path.join(self.curr_dir, "models")
        m1_exists = os.path.exists(os.path.join(models_dir, "YOLOLens1.onnx"))
        m2_exists = os.path.exists(os.path.join(models_dir, "YOLOLens2.onnx"))
        green, red = "color: #2ecc71; font-size: 15pt;", "color: #e74c3c; font-size: 15pt;"
        self.labelModel1Status.setStyleSheet(green if m1_exists else red)
        self.labelModel2Status.setStyleSheet(green if m2_exists else red)
