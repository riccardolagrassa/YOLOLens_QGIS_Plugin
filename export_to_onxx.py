import sys
import torch
import os
from onnxsim import simplify
import onnx

# 1. Setup paths
plugin_dir = os.path.dirname(__file__)
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)
sys.path.append(os.path.abspath("./YoloLensv8"))
model_path = os.path.join(os.path.dirname(__file__), "yololens2.pt")
onnx_path = os.path.join(os.path.dirname(__file__), "YOLOLens2.onnx")

# 2. Load your model
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
model.float().eval()

# 3. Create dummy input (The size here is just a template)
dummy_input = torch.randn(1, 3, 416, 416)

# 4. Export with FULL DYNAMIC AXES
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['sr_out', 'outSR_calibrated', 'yolo_out'],
    # CRITICAL: Added indices 2 and 3 for dynamic Height and Width
    dynamic_axes={
        'input':    {0: 'batch_size', 2: 'height', 3: 'width'},
        'sr_out':   {0: 'batch_size', 2: 'sr_height', 3: 'sr_width'},
        'outSR_calibrated': {0: 'batch_size', 2: 'sr_height', 3: 'sr_width'},
        'yolo_out': {0: 'batch_size', 1: 'classes_4parameters', 2: 'num_anchors'}
    }
)

model_onnx = onnx.load(onnx_path)
dynamic_input_shapes = {"input": [1, 3, 416, 416]}
model_simp, check = simplify(model_onnx, test_input_shapes=dynamic_input_shapes)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_path)
print(f"Success! Model re-exported with dynamic resolution to {onnx_path}")