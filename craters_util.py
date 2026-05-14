import torch
import numpy as np
import torch.nn.functional as F
import torchvision


class CratersUtil:
    def __init__(self):
        """Utility class for crater detection processing."""
        pass

    def non_max_suppression(
            self,
            prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nc=0,
            max_time_img=0.05,
            max_nms=30000,
            max_wh=7680,
    ):
        # Checks
        assert 0 <= conf_thres <= 1
        assert 0 <= iou_thres <= 1

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        device = prediction.device
        mps = 'mps' in device.type
        if mps:
            prediction = prediction.cpu()

        bs = prediction.shape[0]
        nc = nc or (prediction.shape[1] - 4)
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc
        xc = prediction[:, 4:mi].amax(1) > conf_thres

        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):
            x = x.transpose(0, -1)[xc[xi]]

            if not x.shape[0]:
                continue

            box, cls, mask = x.split((4, nc, nm), 1)
            # Call internal method using self
            box = self.xywh2xyxy(box)

            if multi_label:
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            n = x.shape[0]
            if not n:
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            i = i[:max_det]

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)

        return output

    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def xyxy2xywh(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]
        return y

    def box_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def box_iou(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        return inter / (self.box_area(box1.T)[:, None] + self.box_area(box2.T) - inter + eps)

    def pix2coord(self, x, y, cdim, imgdim, origin="upper"):
        cx = (x / imgdim[0]) * (cdim[1] - cdim[0]) + cdim[0]
        if origin == "lower":
            cy = (y / imgdim[1]) * (cdim[3] - cdim[2]) + cdim[2]
        else:
            cy = cdim[3] - (y / imgdim[1]) * (cdim[3] - cdim[2])
        return cx, cy

    def translate_pix2coord(self, pred, crater_coords_locations, single_cdim, confidence, offset, full_offset_list,
                            single_sp_res):
        MperPixel = single_sp_res / 1000
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if len(pred) > 0:
            boxes_xywh_conf = pred[:, 2:7]
            for box_xywhc in boxes_xywh_conf.tolist():
                # Tycho logic (retained from your original)
                if (single_sp_res == 21 * 4 and (box_xywhc[2] * MperPixel + box_xywhc[3] * MperPixel) / 2 > 0.15) or \
                        (single_sp_res == 21 * 2 and (
                                box_xywhc[2] * MperPixel + box_xywhc[3] * MperPixel) / 2 > 0.09) or \
                        (single_sp_res == 21):

                    if box_xywhc[-1] > confidence:
                        x, y = box_xywhc[0], box_xywhc[1]
                        if offset > 0:
                            offset_row = full_offset_list[int(round(x))]
                            x = 416 * ((x - offset_row) / ((416 - offset_row) - (offset_row)))

                        cx, cy = self.pix2coord(x, y, single_cdim, (416, 416), origin="upper")
                        crater_coords_locations.loc[len(crater_coords_locations)] = [
                            cx, cy, (box_xywhc[2]) * MperPixel, (box_xywhc[3]) * MperPixel,
                            box_xywhc[-1], single_cdim[0], single_cdim[1], single_cdim[2], single_cdim[3]
                        ]

    def sample_rect(self, dtm, x1, x2, y1, y2):
        """Helper for morphometric sampling. Safely expands boundaries to catch pixels."""
        if dtm is None or dtm.size == 0:
            return np.array([np.nan], dtype=np.float32)

        # Use floor and ceil to guarantee the bounding window has width/height >= 1
        x1 = int(max(0, np.floor(x1)))
        x2 = int(min(dtm.shape[1], np.ceil(x2)))
        y1 = int(max(0, np.floor(y1)))
        y2 = int(min(dtm.shape[0], np.ceil(y2)))

        if x1 >= x2 or y1 >= y2:
            return np.array([np.nan], dtype=np.float32)

        sliced = dtm[y1:y2, x1:x2]
        if sliced.size == 0:
            return np.array([np.nan], dtype=np.float32)
        return sliced

    def safe_nanmax(self, arr, default=-9999.0):
        """Returns nanmax of array safely, falls back to default if all NaN/empty."""
        clean_arr = arr[~np.isnan(arr)]
        if clean_arr.size == 0:
            return default
        return float(np.max(clean_arr))

    def safe_nanmin(self, arr, default=-9999.0):
        """Returns nanmin of array safely, falls back to default if all NaN/empty."""
        clean_arr = arr[~np.isnan(arr)]
        if clean_arr.size == 0:
            return default
        return float(np.min(clean_arr))

    def process_crater_pixel(self, row, dtm_patch, spatial_res):
        """
        Compute morphometric parameters safely for a crater in pixel coordinates.
        Uses fallback rules to prevent All-NaN slice runtime warnings/errors.
        """
        # Set standardized default values in case computations fail
        result = {
            'Elevation_Center': -9999.0, 'Elevation_Peak': -9999.0, 'E_Rim_Right': -9999.0,
            'E_Rim_Left': -9999.0, 'E_Rim_Top': -9999.0, 'E_Rim_Bottom': -9999.0,
            'avg_Elevation': -9999.0, 'Depth_e_East-Center': -9999.0, 'Depth_e_West-Center': -9999.0,
            'Depth_e_North-Center': -9999.0, 'Depth_e_South-Center': -9999.0, 'd/D': -9999.0
        }

        if dtm_patch is None:
            return result

        try:
            xc, yc = row['x'], row['y']
            w, h = row['w'], row['h']
            D_px = ((w + h) / 2)
            if D_px <= 0:
                return result

            xr, xl = xc + w / 2, xc - w / 2
            yt, yb = yc - h / 2, yc + h / 2

            neighbour = int((1 / 6 * D_px))
            vals_center = self.sample_rect(dtm_patch, xc - neighbour // 2, xc + neighbour // 2, yc - neighbour // 2,
                                           yc + neighbour // 2)

            # Use safe wrapper methods to prevent nan warnings
            depth_center = self.safe_nanmin(vals_center)
            max_peak = self.safe_nanmax(vals_center)

            rim_band = int(1 / 4 * D_px)
            depth_rim_right = self.safe_nanmax(
                self.sample_rect(dtm_patch, xr, xr + rim_band / 2, yc - rim_band / 2, yc + rim_band / 2))
            depth_rim_left = self.safe_nanmax(
                self.sample_rect(dtm_patch, xl - rim_band / 2, xl, yc - rim_band / 2, yc + rim_band / 2))
            depth_rim_top = self.safe_nanmax(
                self.sample_rect(dtm_patch, xc - rim_band / 2, xc + rim_band / 2, yt - rim_band / 2, yt))
            depth_rim_bottom = self.safe_nanmax(
                self.sample_rect(dtm_patch, xc - rim_band / 2, xc + rim_band / 2, yb, yb + rim_band / 2))

            # Compile valid values for the rim to calculate average elevation
            rim_vals = [val for val in [depth_rim_right, depth_rim_left, depth_rim_top, depth_rim_bottom] if
                        val != -9999.0]

            if depth_center != -9999.0:
                avg_rim = float(np.mean(rim_vals))

                result.update({
                    'Elevation_Center': float(depth_center),
                    'Elevation_Peak': float(max_peak),
                    'E_Rim_Right': float(depth_rim_right),
                    'E_Rim_Left': float(depth_rim_left),
                    'E_Rim_Top': float(depth_rim_top),
                    'E_Rim_Bottom': float(depth_rim_bottom),
                    'avg_Elevation': avg_rim,
                    'Depth_e_East-Center': float(
                        depth_rim_right - depth_center) if depth_rim_right != -9999.0 else -9999.0,
                    'Depth_e_West-Center': float(
                        depth_rim_left - depth_center) if depth_rim_left != -9999.0 else -9999.0,
                    'Depth_e_North-Center': float(
                        depth_rim_top - depth_center) if depth_rim_top != -9999.0 else -9999.0,
                    'Depth_e_South-Center': float(
                        depth_rim_bottom - depth_center) if depth_rim_bottom != -9999.0 else -9999.0,
                    'd/D': float((avg_rim - depth_center) / (D_px * spatial_res))
                })
        except Exception:
            pass

        return result
