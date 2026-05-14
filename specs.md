# YOLOLens QGIS Plugin — Technical Guide

## Overview

YOLOLens processes large planetary rasters using a tiled deep learning inference pipeline based on ONNX YOLO models integrated inside QGIS.

The system supports:

- Optical crater detection
- Multimodal crater detection (Optical + DTM + Hillshade)
- Super-resolution reconstruction
- Morphometric parameter extraction
- GIS visualization

---

# Global Pipeline

```text
INPUT RASTER
    │
    ├──► PREPROCESSING
    │       ├─ normalization
    │       └─ hillshade generation
    │
    ├──► SLIDING WINDOW INFERENCE
    │       ├─ tile extraction
    │       ├─ overlap handling
    │       └─ zero padding
    │
    ├──► ONNX YOLO MODEL
    │
    ├──► LOCAL NMS
    │
    ├──► GLOBAL IoU DEDUPLICATION
    │
    ├──► MORPHOMETRIC ANALYSIS
    │
    └──► QGIS OUTPUT LAYERS
```

---

# 1. Preprocessing

## Model 1 — Optical Only

The optical raster patch is normalized and replicated into 3 channels.

```python
input_data = np.stack([patch / max_v] * 3, axis=0)
```

---

## Model 2 — Multimodal Input

Model 2 uses:

- Optical raster
- DTM
- Hillshade

Input tensor:

```text
Channel 1 → Optical image
Channel 2 → Normalized DTM
Channel 3 → Hillshade
```

---

## DTM Normalization

```python
MOON_MIN = -9178.0
MOON_MAX = 10786.0
```

```python
dtm_norm = (dtm - MOON_MIN) / (MOON_MAX - MOON_MIN)
```

The values are clipped into:

```text
[0, 1]
```

---

## Hillshade Generation

Hillshade is computed from DTM gradients:

```python
np.gradient(dtm.astype('float32'))
```

using illumination parameters:

```text
Azimuth  = 315°
Altitude = 45°
```

This enhances crater morphology.

---

# 2. Sliding Window Inference

Large planetary rasters cannot be processed simultaneously due to:

- GPU memory limitations
- ONNX tensor constraints
- computational cost

YOLOLens therefore uses overlapping sliding windows.

---

## Tile Configuration

```python
tile_size = 256
overlap   = 128
stride    = 128
```

---

## Window Structure

```text
|------256------|
      |------256------|
```

Overlap guarantees that craters near borders remain fully visible.

---

## Sliding Iteration

```python
for y_off in y_steps:
    for x_off in x_steps:
```

Each tile is independently processed.

---

## Border Padding

Edge tiles smaller than `256×256` are padded:

```python
padded = np.zeros((3, tile_size, tile_size))
```

This guarantees fixed ONNX input dimensions.

---

# 3. ONNX Inference

Inference is executed using:

```python
session.run(...)
```

---

## Model 1 Outputs

```text
yolo_out
```

---

## Model 2 Outputs

```text
sr_out
outSR_calibrated
yolo_out
```

Where:

- `sr_out` → super-resolved optical image
- `outSR_calibrated` → super-resolved DTM
- `yolo_out` → crater detections

---

# 4. Super-Resolution Reconstruction

Model 2 reconstructs full-resolution mosaics from overlapping tiles.

---

## Reconstruction Pipeline

```text
SR tile
   │
   ├──► Hann weighting
   │
   └──► accumulation into global raster
```

---

## Hann Window Blending

The plugin generates a 2D Hann window:

```python
np.hanning(H)
```

Purpose:

```text
Without Hann → visible seams
With Hann    → smooth reconstruction
```

---

## Weighted Accumulation

```python
recon += sr_patch * hann_window
weight_mask += hann_window
```

Final normalization:

```python
final = recon / (weight_mask + 1e-8)
```

---

# 5. Local Detection Filtering (NMS)

YOLO frequently predicts multiple boxes for the same crater.

Example:

```text
Box A → 0.92
Box B → 0.88
Box C → 0.81
```

---

## Non-Maximum Suppression

The plugin applies:

```python
torchvision.ops.nms(...)
```

using:

```python
iou_thres = 0.45
```

Logic:

```text
If IoU > 0.45
→ keep highest-confidence box
→ suppress overlapping detections
```

---

# 6. Global Coordinate Projection

Tile detections are projected back into global raster coordinates:

```python
'x': (x / sr_factor) + x_idx
'y': (y / sr_factor) + y_idx
```

---

# 7. Global IoU Deduplication

Even after local NMS, duplicate detections may still exist across overlapping tiles.

```text
Tile A detects crater
Tile B detects same crater
```

---

## Deduplication Pipeline

```text
Detections
    │
    ├──► Polygon conversion
    │
    ├──► Spatial indexing
    │
    ├──► IoU comparison
    │
    └──► Cluster merging
```

---

## Polygon Construction

Each crater box becomes a Shapely polygon:

```python
Polygon([
    (x1, y1),
    (x2, y1),
    (x2, y2),
    (x1, y2)
])
```

---

## IoU Filtering

```python
IoU = intersection / union
```

Threshold:

```python
IoU >= 0.6
```

Meaning:

```text
detections belong to the same crater
```

---

## Best Detection Selection

For each cluster:

```python
best = sub.loc[sub['conf'].idxmax()]
```

The highest-confidence crater is retained.

---

# 8. Morphometric Analysis

Available only in Model 2.

Computed parameters include:

- crater center elevation
- rim elevations
- peak elevation
- crater depth
- d/D ratio

---

## Safe Sampling

The plugin safely extracts DTM regions using:

```python
sample_rect()
```

---

## NaN Handling

Planetary DTMs may contain:

- invalid pixels
- gaps
- missing values

Safe wrappers:

```python
safe_nanmin()
safe_nanmax()
```

prevent runtime failures.

---

## d/D Ratio

```text
d/D = crater_depth / crater_diameter
```

Implemented as:

```python
(avg_rim - depth_center) / (D_px * spatial_res)
```

---

# 9. Final QGIS Outputs

The plugin exports:

## Vector Layer

```text
Detected crater points
```

including:

- confidence
- pixel diameter
- geographic diameter
- morphometric parameters

---

## Raster Layers (Model 2)

```text
Super-resolved DTM
Super-resolved Optical Mosaic
```

These layers are automatically loaded into QGIS.

---

# Compact Workflow Diagram

```text
INPUT RASTER
    ↓
TILE SPLIT (256, overlap 128)
    ↓
NORMALIZATION / HILLSHADE
    ↓
ONNX INFERENCE
    ↓
LOCAL NMS (IoU 0.45)
    ↓
GLOBAL DEDUP (IoU 0.60)
    ↓
MORPHOMETRY
    ↓
QGIS EXPORT
```

---

# Core Technical Concepts

| Component | Purpose |
|---|---|
| Sliding Window | Process massive rasters |
| Tile Overlap | Prevent crater truncation |
| Hann Window | Seamless SR mosaics |
| Local NMS | Remove duplicate local boxes |
| Global IoU Filtering | Remove inter-tile duplicates |
| Morphometry | Extract crater topology |
| QGIS Export | GIS visualization |

---

# Summary

YOLOLens combines:

- tiled ONNX inference
- super-resolution reconstruction
- multimodal fusion
- geometric IoU filtering
- morphometric crater analysis

into a scalable planetary GIS workflow optimized for large-scale crater detection.
````
