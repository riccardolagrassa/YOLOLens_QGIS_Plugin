#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IoU-based crater deduplication with transitive merging (connected components)
Author: Riccardo La Grassa (corrected version)
"""

import time
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from multiprocessing import Pool, cpu_count
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. Read craters and build pixel-space polygons
# ============================================================

def read_craters_pixel(filename):
    """Read crater CSV and build rectangular polygons in pixel space."""

    cr = pd.read_csv(filename, sep=r"\s+")

    polys = []
    for r in cr.itertuples():
        hw = r.w / 2
        hh = r.h / 2
        polys.append(Polygon([
            (r.x - hw, r.y - hh),
            (r.x + hw, r.y - hh),
            (r.x + hw, r.y + hh),
            (r.x - hw, r.y + hh),
        ]))

    cr = gpd.GeoDataFrame(cr, geometry=polys)
    cr = cr.reset_index(drop=True)  # important for spatial index
    return cr


# ============================================================
# 2. Compute IoU between polygon A and B
# ============================================================

def iou(poly1, poly2):
    inter = poly1.intersection(poly2).area
    if inter == 0.0:
        return 0.0
    union = poly1.union(poly2).area
    return inter / union


# ============================================================
# 3. Build adjacency lists for IoU graph (single-threaded)
# ============================================================

def build_iou_graph(craters, iou_thresh=0.6):
    """
    Construct graph G: if IoU(i,j) >= threshold → edge between nodes.
    Spatial index reduces comparisons.
    """

    n = len(craters)
    sindex = craters.sindex
    adjacency = [[] for _ in range(n)]

    for i, row in craters.iterrows():

        # candidate neighbors via bounding box intersection
        cand = list(sindex.intersection(row.geometry.bounds))

        for j in cand:
            if j <= i:
                continue  # avoid double checks

            if row.geometry.intersects(craters.geometry[j]):
                if iou(row.geometry, craters.geometry[j]) >= iou_thresh:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

    return adjacency


# ============================================================
# 4. Connected Components (transitive closure)
# ============================================================

def connected_components(adjacency):
    """
    Standard DFS-based connected components.
    Returns list of lists, each sublist = indices belonging to same cluster.
    """

    n = len(adjacency)
    visited = [False]*n
    clusters = []

    for i in range(n):
        if not visited[i]:
            stack = [i]
            comp = []
            visited[i] = True

            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adjacency[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            clusters.append(comp)

    return clusters


# ============================================================
# 5. Merge polygons of a cluster (parallelizable helper)
# ============================================================

def merge_cluster(args):
    """
    Merge a cluster of duplicate detections.
    Preserve morphometric parameters from the highest confidence detection.
    """
    subcluster, craters = args

    geoms = [craters.geometry[i] for i in subcluster]
    merged_poly = unary_union(geoms)

    min_x, min_y, max_x, max_y = merged_poly.bounds
    x_final = (min_x + max_x) / 2
    y_final = (min_y + max_y) / 2
    w_final = max_x - min_x
    h_final = max_y - min_y

    # keep all attributes from the highest confidence detection
    subdf = craters.loc[subcluster]
    best = subdf.loc[subdf.conf.idxmax()]

    # morphometric columns to copy
    morphometric_cols = [
        'Elevation_Center', 'Elevation_Peak', 'E_Rim_Right', 'E_Rim_Left',
        'E_Rim_Top', 'E_Rim_Bottom', 'avg_Elevation',
        'Depth_e_East-Center', 'Depth_e_West-Center',
        'Depth_e_North-Center', 'Depth_e_South-Center', 'd/D'
    ]

    result = {
        "x": x_final,
        "y": y_final,
        "w": w_final,
        "h": h_final,
        "lon": best.lon,
        "lat": best.lat,
        "w_geo": best.w_geo,
        "h_geo": best.h_geo,
        "conf": float(subdf.conf.max()),
        "geometry": merged_poly
    }

    # copy morphometric parameters from best
    for col in morphometric_cols:
        if col in best:
            result[col] = best[col]

    return result



# ============================================================
# 6. Full pipeline
# ============================================================

def filter_craters(csv_path, output_path, iou_thresh=0.6, num_processes=None):
    """
    Reads craters → builds IoU graph → clusters → parallel geometric union → saves.
    Deterministic, correct, race-free.
    """

    t0 = time.time()
    print("Loading crater file…")
    cr = read_craters_pixel(csv_path)

    print("Building IoU graph…")
    adjacency = build_iou_graph(cr, iou_thresh=iou_thresh)

    print("Computing connected components…")
    clusters = connected_components(adjacency)
    print(f"Found {len(clusters)} clusters.")

    # prepare tasks
    tasks = [(cl, cr) for cl in clusters]

    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)

    print(f"Merging clusters using {num_processes} processes…")
    with Pool(num_processes) as pool:
        merged_rows = pool.map(merge_cluster, tasks)

    final_df = gpd.GeoDataFrame(merged_rows)
    morphometric_cols = [
                        "Elevation_Center",
                        "Elevation_Peak",
                        "E_Rim_Right",
                        "E_Rim_Left",
                        "E_Rim_Top",
                        "E_Rim_Bottom",
                        "avg_Elevation",
                        "Depth_e_East-Center",
                        "Depth_e_West-Center",
                        "Depth_e_North-Center",
                        "Depth_e_South-Center",
                        "d/D"
                        ]
    cols_to_save = ["x", "y", "w", "h", "lon", "lat", "w_geo", "h_geo", "conf"] + morphometric_cols

    final_df[cols_to_save].to_csv(output_path, index=False)

    print(f"Finished in {time.time() - t0:.2f} seconds.")
    print(f"Output saved to: {output_path}")

    return final_df
