# -*- coding: utf-8 -*-
import os
import torch
import sys
import time
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import gc

# Add the plugin directory to the Python path
plugin_dir = os.path.dirname(__file__)
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

from osgeo import gdal
from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.PyQt.QtGui import QIcon, QColor
from qgis.PyQt.QtWidgets import QAction, QApplication
from qgis.core import (QgsVectorLayer, QgsField, QgsFeature, QgsGeometry,
                       QgsPointXY, QgsProject, QgsMarkerSymbol,
                       QgsSimpleMarkerSymbolLayer, QgsProperty,
                       QgsSingleSymbolRenderer, QgsUnitTypes, QgsMessageLog, QgsRasterLayer)

from .craters_util import CratersUtil
from .resources import *
from .crater_detector_dialog import CraterDetectorDialog


class CraterDetector:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = self.tr(u'&YOLOLens')
        self.onnx_session = None
        self.active_device = ""
        self.available_devices = "None"
        self.utils = CratersUtil()
        self.current_model_path = ""
        self._hann_cache = {}
        self.detect_hardware()

        try:
            self.get_onnx_session()
        except Exception as e:
            QgsMessageLog.logMessage(f"YOLOLens Init Failed: {e}", "YOLOLens")

    def tr(self, message):
        return QCoreApplication.translate('CraterDetector', message)

    def detect_hardware(self):
        """Detects hardware capabilities without needing a model file loaded."""
        try:
            import onnxruntime as ort
            # Get all providers supported by the installed ONNX Runtime
            all_providers = ort.get_available_providers()
            readable = []
            if 'CUDAExecutionProvider' in all_providers: readable.append("GPU (CUDA)")
            if 'CPUExecutionProvider' in all_providers: readable.append("CPU")

            self.available_devices = " & ".join(readable) if readable else "CPU Only"
            # Default active device to the best available
            self.active_device = "GPU (CUDA)" if 'CUDAExecutionProvider' in all_providers else "CPU"
        except Exception as e:
            self.available_devices = "Error detecting hardware"
            QgsMessageLog.logMessage(f"YOLOLens Hardware Detect Failed: {e}", "YOLOLens")

    def add_action(self, icon_path, text, callback, enabled_flag=True,
                   add_to_menu=True, add_to_toolbar=True, status_tip=None,
                   whats_this=None, parent=None):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)
        if status_tip: action.setStatusTip(status_tip)
        if whats_this: action.setWhatsThis(whats_this)
        if add_to_toolbar: self.iface.addToolBarIcon(action)
        if add_to_menu: self.iface.addPluginToMenu(self.menu, action)
        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = ':/plugins/crater_detector/yololens.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Detect Craters'),
            callback=self.run,
            parent=self.iface.mainWindow())

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.tr(u'&YOLOLens'), action)
            self.iface.removeToolBarIcon(action)

    def get_onnx_session(self, model_name="model.onnx"):
        if self.onnx_session is None or os.path.basename(self.current_model_path) != model_name:
            self.onnx_session = None
            import onnxruntime as ort
            try:
                ort.preload_dlls()
            except:
                pass

            model_path = os.path.join(self.plugin_dir, "models", model_name)
            if not os.path.exists(model_path):
                model_path = os.path.join(self.plugin_dir, model_name)

            if not os.path.exists(model_path):
                return None

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            self.current_model_path = model_path
            actual_providers = self.onnx_session.get_providers()
            readable_providers = []
            if 'CUDAExecutionProvider' in actual_providers: readable_providers.append("GPU (CUDA)")
            if 'CPUExecutionProvider' in actual_providers: readable_providers.append("CPU")
            self.available_devices = ", ".join(readable_providers)
            self.active_device = "GPU (CUDA)" if actual_providers[0] == 'CUDAExecutionProvider' else "CPU"

        return self.onnx_session

    def run(self):
        self.dlg = CraterDetectorDialog()

        def check_specific_models():
            models_dir = os.path.join(self.plugin_dir, "models")
            m1_exists = os.path.exists(os.path.join(models_dir, "YOLOLens1.onnx"))
            m2_exists = os.path.exists(os.path.join(models_dir, "YOLOLens2.onnx"))
            green_style, red_style = "color: #2ecc71; font-size: 15pt;", "color: #e74c3c; font-size: 15pt;"
            self.dlg.labelModel1Status.setStyleSheet(green_style if m1_exists else red_style)
            self.dlg.labelModel2Status.setStyleSheet(green_style if m2_exists else red_style)

        check_specific_models()
        self.dlg.labelDevice.setText(self.available_devices)
        self.dlg.labelStatus.setText("Ready")
        self.dlg.pushButton.clicked.connect(self.process_crater_detection)
        self.dlg.finished.connect(self.clear_model_memory)
        self.dlg.show()
        self.dlg.exec_()

    def clear_model_memory(self):
        if self.onnx_session is not None:
            self.onnx_session = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            QgsMessageLog.logMessage("YOLOLens: Model cleared from memory.", "YOLOLens")

    def process_crater_detection(self):
        layer = self.dlg.mMapLayerComboBox_Optical.currentLayer()
        conf_threshold = self.dlg.doubleSpinBoxConf.value()
        use_multichannel = self.dlg.radioModel2.isChecked()

        dtm_layer = None
        if use_multichannel:
            dtm_layer = self.dlg.mMapLayerComboBox_DTM.currentLayer()
            if not dtm_layer or not layer:
                self.iface.messageBar().pushMessage("Error", "Both layers required for Model 2.", level=4)
                return

        self.dlg.labelStatus.setText(f"RUNNING ON {self.active_device}...")
        self.dlg.labelStatus.setStyleSheet("color: #e67e22; font-weight: bold;")
        self.dlg.pushButton.setEnabled(False)
        QApplication.processEvents()

        try:
            start_time = time.perf_counter()
            model_file = "YOLOLens2.onnx" if use_multichannel else "YOLOLens1.onnx"
            session = self.get_onnx_session(model_file)
            if session is None:
                raise FileNotFoundError(f"Model file '{model_file}' could not be loaded.")

            self.dlg.labelStatus.setText(f"INFERENCE ({self.active_device})")
            self.run_inference(layer, dtm_layer, conf_threshold, session)
            end_time = time.perf_counter()
            self.dlg.labelTime.setText(f"{(end_time - start_time) * 1000:.2f} ms")
            self.dlg.labelStatus.setText(f"SUCCESS ({self.active_device})")
            self.dlg.labelStatus.setStyleSheet("color: #27ae60; font-weight: bold;")
        except Exception as e:
            self.dlg.labelStatus.setText("FAILED")
            self.dlg.labelStatus.setStyleSheet("color: #c0392b; font-weight: bold;")
            self.iface.messageBar().pushMessage("Error", str(e), level=4)
        finally:
            self.dlg.pushButton.setEnabled(True)

    def generate_hillshade(self, dtm):
        y, x = np.gradient(dtm.astype('float32'), 7.0, 7.0)
        az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
        slope = np.arctan(np.sqrt(x ** 2 + y ** 2))
        aspect = np.arctan2(-x, y)
        return np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)

    def normalize_input(self, vis, dtm, hill):
        MOON_MIN, MOON_MAX = -9178.0, 10786.0
        vis_norm = (vis - vis.min()) / (vis.max() - vis.min() + 1e-7)
        dtm_norm = np.zeros_like(dtm)
        mask = (dtm != -9999) & np.isfinite(dtm)
        dtm_norm[mask] = (dtm[mask] - MOON_MIN) / (MOON_MAX - MOON_MIN)
        dtm_norm = np.clip(dtm_norm, 0.0, 1.0)
        hill_norm = np.nan_to_num(hill, nan=0.0)
        hill_norm = np.clip(hill_norm, 0.0, 1.0)
        return np.stack([vis_norm, dtm_norm, hill_norm], axis=0).astype(np.float32)

    def get_hann_window(self, H, W):
        key = (H, W)
        if key in self._hann_cache: return self._hann_cache[key]
        wy, wx = np.hanning(H).astype(np.float32), np.hanning(W).astype(np.float32)
        w2d = np.outer(wy, wx)
        w2d /= (w2d.max() + 1e-7)
        self._hann_cache[key] = w2d
        return w2d

    def get_local_pred(self, pred, confidence, local_craters_list, y_idx, x_idx, dtm_patch, spatial_res, use_model2, sr_factor):
        x1, y1, x2, y2, conf, cls = pred
        if conf < confidence: return
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2, y1 + h / 2

        if use_model2 and dtm_patch is not None:
            # Only extract morphometric parameters if model 2 is chosen
            morph = self.utils.process_crater_pixel(row={'x': x, 'y': y, 'w': w, 'h': h},
                                                    dtm_patch=dtm_patch, spatial_res=spatial_res)
        else:
            # Placeholder parameters for Model 1
            morph = {
                'Elevation_Center': -9999.0, 'Elevation_Peak': -9999.0, 'E_Rim_Right': -9999.0,
                'E_Rim_Left': -9999.0, 'E_Rim_Top': -9999.0, 'E_Rim_Bottom': -9999.0,
                'avg_Elevation': -9999.0, 'Depth_e_East-Center': -9999.0, 'Depth_e_West-Center': -9999.0,
                'Depth_e_North-Center': -9999.0, 'Depth_e_South-Center': -9999.0, 'd/D': -9999.0
            }

        local_craters_list.append({
            'x': (x / sr_factor) + x_idx,
            'y': (y / sr_factor) + y_idx,
            'w': w / sr_factor,
            'h': h / sr_factor,
            'conf': conf,
            **morph
        })

    def run_inference(self, layer, dtm_layer, conf_threshold, session):
        use_model2 = dtm_layer is not None
        ds = gdal.Open(layer.source())
        gt = ds.GetGeoTransform()
        img_w, img_h = ds.RasterXSize, ds.RasterYSize
        extent = layer.extent()
        res_x, res_y = extent.width() / img_w, extent.height() / img_h
        spatial_res = (abs(gt[1]) + abs(gt[5])) / 2

        band_dtm = None
        if use_model2:
            ds_dtm = gdal.Open(dtm_layer.source())
            band_dtm = ds_dtm.GetRasterBand(1)

        # Super-Resolution dimensions variables
        sr_factor = 2
        tile_size, overlap = 256, 128
        stride = tile_size - overlap

        # Setup reconstruction arrays only if Model 2 is used
        recon_dtm = None
        recon_sr_optical = None
        weight_mask = None

        if use_model2:
            target_shape = (img_h * sr_factor, img_w * sr_factor)
            recon_dtm = np.zeros(target_shape, dtype=np.float32)
            recon_sr_optical = np.zeros(target_shape, dtype=np.float32)  # NEW
            weight_mask = np.zeros(target_shape, dtype=np.float32)

        all_detections = []

        x_steps, y_steps = range(0, img_w, stride), range(0, img_h, stride)
        total_tiles = len(x_steps) * len(y_steps)
        processed_tiles = 0

        for y_off in y_steps:
            for x_off in x_steps:
                win_w, win_h = min(tile_size, img_w - x_off), min(tile_size, img_h - y_off)
                patch = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_w, win_h).astype(np.float32)

                if use_model2:
                    patch_dtm = band_dtm.ReadAsArray(x_off, y_off, win_w, win_h).astype(np.float32)
                    input_data = self.normalize_input(patch, patch_dtm, self.generate_hillshade(patch_dtm))
                else:
                    max_v = max(patch.max(), 255.0) if patch.max() > 1.001 else 1.0
                    input_data = np.stack([patch / max_v] * 3, axis=0)

                if win_w < tile_size or win_h < tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=np.float32)
                    padded[:, :win_h, :win_w] = input_data
                    input_data = padded

                # Handle output configurations cleanly based on model type
                if use_model2:
                    outputs = session.run(['sr_out', 'outSR_calibrated', 'yolo_out'],
                                          {'input': np.expand_dims(input_data, 0)})

                    # Calculate slice dimensions
                    valid_h_sr, valid_w_sr = win_h * sr_factor, win_w * sr_factor
                    h_w = self.get_hann_window(valid_h_sr, valid_w_sr)

                    y_start, y_end = y_off * sr_factor, y_off * sr_factor + valid_h_sr
                    x_start, x_end = x_off * sr_factor, x_off * sr_factor + valid_w_sr

                    # 1. SR DTM Reconstruction
                    sr_dtm_patch = outputs[1][0, 0][:valid_h_sr, :valid_w_sr]
                    recon_dtm[y_start:y_end, x_start:x_end] += sr_dtm_patch * h_w

                    # 2. SR Optical Reconstruction (NEW)
                    sr_opt_patch = outputs[0][0, 0][:valid_h_sr, :valid_w_sr]
                    recon_sr_optical[y_start:y_end, x_start:x_end] += sr_opt_patch * h_w

                    weight_mask[y_start:y_end, x_start:x_end] += h_w

                    yolo_output = outputs[2]
                    target_dtm_patch = outputs[1][0, 0]
                    calc_res = spatial_res / sr_factor
                else:
                    outputs = session.run(['yolo_out'],
                                          {'input': np.expand_dims(input_data, 0)})
                    yolo_output = outputs[0]
                    target_dtm_patch = None
                    calc_res = spatial_res

                # 2. Crater Detection
                results = self.utils.non_max_suppression(torch.from_numpy(yolo_output), conf_threshold, 0.45)
                if results and results[0] is not None:
                    for det in results[0].cpu().numpy():
                        # YOLO maps relative to 512 patch dimensions, scale down by factor of 2.0
                        lx1, ly1, lx2, ly2 = det[0], det[1], det[2], det[3]

                        # Guard to ensure bounding box centers land inside the unpadded boundary area
                        if lx1 < (win_w*sr_factor) and ly1 < (win_h*sr_factor):
                            self.get_local_pred([lx1, ly1, lx2, ly2, det[4], det[5]], conf_threshold,
                                                all_detections, y_off, x_off, target_dtm_patch, calc_res, use_model2, sr_factor)

                processed_tiles += 1
                self.dlg.progressBar.setValue(int((processed_tiles / total_tiles) * 100))
                QApplication.processEvents()

        # Finalize DTM Raster layer ONLY for Model 2
        if use_model2:
            recon_dtm /= (weight_mask + 1e-8)
            recon_sr_optical /= (weight_mask + 1e-8)
            self.visualize_raster(recon_dtm, layer, gt, f"YOLOLens_SR_DTM_{conf_threshold}")
            self.visualize_raster(recon_sr_optical, layer, gt, f"YOLOLens_SR_Optical_{conf_threshold}")

        if not all_detections:
            self.iface.messageBar().pushMessage("Info", "No craters detected.", level=3)
            return

        final_craters = self.apply_deduplication(all_detections, use_model2)
        features = []
        for _, row in final_craters.iterrows():
            px_x, px_y = row['x'], row['y']
            f = QgsFeature()
            f.setGeometry(
                QgsGeometry.fromPointXY(QgsPointXY(extent.xMinimum() + px_x * res_x, extent.yMaximum() - px_y * res_y)))

            # Map parameters dynamically (inserting fallback values for Model 1 detections)
            f.setAttributes([
                float(px_x), float(px_y), float(row['conf']), float(row['w']), float(row['h']),
                float(row['w'] * abs(gt[1])), float(row['h'] * abs(gt[5])),
                float(row.get('Elevation_Center', -9999.0)),
                float(row.get('Elevation_Peak', -9999.0)),
                float(row.get('E_Rim_Right', -9999.0)),
                float(row.get('E_Rim_Left', -9999.0)),
                float(row.get('E_Rim_Top', -9999.0)),
                float(row.get('E_Rim_Bottom', -9999.0)),
                float(row.get('avg_Elevation', -9999.0)),
                float(row.get('Depth_e_East-Center', -9999.0)),
                float(row.get('Depth_e_West-Center', -9999.0)),
                float(row.get('Depth_e_North-Center', -9999.0)),
                float(row.get('Depth_e_South-Center', -9999.0)),
                float(row.get('d/D', -9999.0))
            ])
            features.append(f)

        self.create_output_layer(layer, features, conf_threshold)

    def visualize_raster(self, data, layer, gt, layer_name):
        """Generalized method to visualize SR output rasters."""
        temp_tif = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(temp_tif, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform((gt[0], gt[1] / 2, gt[2], gt[3], gt[4], gt[5] / 2))
        out_ds.SetProjection(gdal.Open(layer.source()).GetProjection())
        out_ds.GetRasterBand(1).WriteArray(data)
        out_ds = None
        rlayer = QgsRasterLayer(temp_tif, layer_name)
        QgsProject.instance().addMapLayer(rlayer)

    def apply_deduplication(self, detections_list, use_model2):
        df = pd.DataFrame(detections_list)
        polys = [Polygon(
            [(r.x - r.w / 2, r.y - r.h / 2), (r.x + r.w / 2, r.y - r.h / 2), (r.x + r.w / 2, r.y + r.h / 2),
             (r.x - r.w / 2, r.y + r.h / 2)]) for r in df.itertuples()]
        gdf = gpd.GeoDataFrame(df, geometry=polys)
        sindex, n = gdf.sindex, len(gdf)
        adj = [[] for _ in range(n)]
        for i, row in gdf.iterrows():
            for j in sindex.intersection(row.geometry.bounds):
                if j > i and row.geometry.intersects(gdf.geometry[j]):
                    p1, p2 = row.geometry, gdf.geometry[j]
                    if (p1.intersection(p2).area / p1.union(p2).area) >= 0.6:
                        adj[i].append(j)
                        adj[j].append(i)

        visited, final_rows = [False] * n, []
        for i in range(n):
            if not visited[i]:
                stack, cluster = [i], []
                visited[i] = True
                while stack:
                    node = stack.pop()
                    cluster.append(node)
                    for nei in adj[node]:
                        if not visited[nei]: visited[nei] = True; stack.append(nei)
                sub = gdf.iloc[cluster]
                best = sub.loc[sub['conf'].idxmax()]
                poly = unary_union(sub.geometry.tolist())
                minx, miny, maxx, maxy = poly.bounds

                # Base dictionary constructed for all model runs
                base_dict = {
                    'x': (minx + maxx) / 2, 'y': (miny + maxy) / 2, 'w': maxx - minx, 'h': maxy - miny,
                    'conf': best['conf']
                }

                # Conditional inclusion of morphometrics during grouping
                if use_model2:
                    base_dict.update({
                        'Elevation_Center': float(best.get('Elevation_Center', -9999.0)),
                        'Elevation_Peak': float(best.get('Elevation_Peak', -9999.0)),
                        'E_Rim_Right': float(best.get('E_Rim_Right', -9999.0)),
                        'E_Rim_Left': float(best.get('E_Rim_Left', -9999.0)),
                        'E_Rim_Top': float(best.get('E_Rim_Top', -9999.0)),
                        'E_Rim_Bottom': float(best.get('E_Rim_Bottom', -9999.0)),
                        'avg_Elevation': float(best.get('avg_Elevation', -9999.0)),
                        'Depth_e_East-Center': float(best.get('Depth_e_East-Center', -9999.0)),
                        'Depth_e_West-Center': float(best.get('Depth_e_West-Center', -9999.0)),
                        'Depth_e_North-Center': float(best.get('Depth_e_North-Center', -9999.0)),
                        'Depth_e_South-Center': float(best.get('Depth_e_South-Center', -9999.0)),
                        'd/D': float(best.get('d/D', -9999.0))
                    })
                else:
                    base_dict.update({
                        'Elevation_Center': -9999, 'Elevation_Peak': -9999,
                        'E_Rim_Right': -9999, 'E_Rim_Left': -9999,
                        'E_Rim_Top': -9999, 'E_Rim_Bottom': -9999,
                        'avg_Elevation': -9999, 'Depth_e_East-Center': -9999,
                        'Depth_e_West-Center': -9999, 'Depth_e_North-Center': -9999,
                        'Depth_e_South-Center': -9999, 'd/D': -9999,
                    })

                final_rows.append(base_dict)

        return pd.DataFrame(final_rows)

    def create_output_layer(self, layer, features, conf):
        vl = QgsVectorLayer("Point", f"YOLOLens_Detections_{conf}", "memory")
        vl.setCrs(layer.crs())
        fields = [QgsField("x", QVariant.Double), QgsField("y", QVariant.Double), QgsField("conf", QVariant.Double),
                  QgsField("W_px", QVariant.Double), QgsField("H_px", QVariant.Double),
                  QgsField("W_geo", QVariant.Double), QgsField("H_geo", QVariant.Double),
                  QgsField("Elevation_Center", QVariant.Double),
                  QgsField("Elevation_Peak", QVariant.Double),
                  QgsField("E_Rim_Right", QVariant.Double),
                  QgsField("E_Rim_L", QVariant.Double),
                  QgsField("E_Rim_T", QVariant.Double),
                  QgsField("E_Rim_B", QVariant.Double),
                  QgsField("Avg_Elev", QVariant.Double),
                  QgsField("D_East_C", QVariant.Double),
                  QgsField("D_West_C", QVariant.Double),
                  QgsField("D_North_C", QVariant.Double),
                  QgsField("D_South_C", QVariant.Double),
                  QgsField("d/D", QVariant.Double)
                  ]
        vl.dataProvider().addAttributes(fields)
        vl.updateFields()
        vl.dataProvider().addFeatures(features)

        symbol_layer = QgsSimpleMarkerSymbolLayer()
        symbol_layer.setShape(QgsSimpleMarkerSymbolLayer.Circle)
        symbol_layer.setSizeUnit(QgsUnitTypes.RenderMapUnits)
        symbol_layer.setColor(QColor(0, 0, 0, 0))
        symbol_layer.setStrokeColor(QColor(255, 0, 0))
        symbol_layer.setDataDefinedProperty(QgsSimpleMarkerSymbolLayer.PropertySize,
                                            QgsProperty.fromExpression('( \"W_geo\" + \"H_geo\" ) / 2'))
        vl.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol([symbol_layer])))
        QgsProject.instance().addMapLayer(vl)
        self.iface.messageBar().pushMessage("Success", f"Detected {len(features)} craters.", level=3)