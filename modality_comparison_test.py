import re
import sys
import argparse
import math
import numpy as np
import pandas as pd
import pathlib
import utils.load_and_write as lw

from Curve import Curve


import time


def process_folder(_folder_path):

    """ ---------- Data Loading ---------- """
    folder_path = pathlib.Path(_folder_path)
    curves_filenames = [f.name for f in folder_path.iterdir() if f.is_file() and not f.name.startswith('._')]

    # separate the modalities fp (fundus photography), ir (infrared photography)
    fp_filenames = [f for f in curves_filenames if re.search(r'_fp(?:_|\.|$)', f)]
    ir_filenames = [f for f in curves_filenames if re.search(r'_ir(?:_|\.|$)', f)]
    fp_filenames = sorted(fp_filenames)
    ir_filenames = sorted(ir_filenames)

    fp_raw_curves = lw.load_curves_from_filenames(_folder_path, fp_filenames)
    ir_raw_curves = lw.load_curves_from_filenames(_folder_path, ir_filenames)

    """ ---------- End Data Loading ---------- """

    filenames = []
    tortuosity_metrics = []
    for fp_raw_curve, ir_raw_curve, fp_filename, ir_filename in zip(fp_raw_curves, ir_raw_curves, fp_filenames, ir_filenames):
        if fp_filename.replace('_fp.txt', '') != ir_filename.replace('_ir.txt', ''):
                        raise ValueError("Curves do not match: {} and {}".format(fp_filename, ir_filename))
        # fp_raw_curve += 1  # add 1 to be MATLAB compatible, (1 based indexing)
        # ir_raw_curve += 1  # add 1 to be MATLAB compatible, (1 based indexing)

        fp_raster_curve = Curve(fp_raw_curve[:, 0], fp_raw_curve[:, 1])
        ir_raster_curve = Curve(ir_raw_curve[:, 0], ir_raw_curve[:, 1])

        # --- proposed heuristics for smoothing and resampling ---
        # --- you can change with yours ---
        fp_Lc = fp_raster_curve.arclength()
        ir_Lc = ir_raster_curve.arclength()
        fp_num_points = math.ceil(len(fp_raster_curve) * 0.25)
        ir_num_points = math.ceil(len(ir_raster_curve) * 0.25)
        fp_smoothness = 8 / fp_Lc
        ir_smoothness = 8 / ir_Lc

        

        # --- smooth and resample the curve        
        fp_smoothed_curve = fp_raster_curve.smooth("cubic_spline", smooth=fp_smoothness, num_points=len(fp_raw_curve))
        ir_smoothed_curve = ir_raster_curve.smooth("cubic_spline", smooth=ir_smoothness, num_points=len(ir_raw_curve))       
        fp_param_curve = fp_smoothed_curve.parametrize("scc", num_points=ir_num_points)
        ir_param_curve = ir_smoothed_curve.parametrize("scc", num_points=ir_num_points)        

        fp_tm = compute_tortuosities(fp_param_curve)
        ir_tm = compute_tortuosities(ir_param_curve)

        tm = sum(zip(fp_tm, ir_tm), ())
        tortuosity_metrics.append(tm)
        filenames.append(fp_filename.replace('_fp.txt', ''))

    return tortuosity_metrics, filenames



def compute_tortuosities(curve):
    DM = curve.tortuosity("DM")
    T_2 = curve.tortuosity("total_curvature", smooth=1.0)
    T_3 = curve.tortuosity("tau_3", smooth=1.0)
    T_5 = curve.tortuosity("tau_5", smooth=1.0)
    ASDC = curve.tortuosity("ASDC", smooth=1.0)
    ICM = curve.tortuosity("ICM", smooth=1.0)
    TD = curve.tortuosity("TD", smooth=1.0)
    SOAM = curve.tortuosity("SOAM")
    SCC = curve.tortuosity("SCC")
    ESCC = curve.tortuosity("ESCC")

    return [DM, T_2, T_3, T_5, ASDC, ICM, TD, SOAM, SCC, ESCC]


def main():
    parser = argparse.ArgumentParser(description='Compute tortuosity metrics for a given curve')
    parser.add_argument('path', help='Path to folder containing the curve txt files')
    parser.add_argument('output_file', help='Path to the output file')
    args = parser.parse_args()

    
    tortuosity_metrics, filenames = process_folder(args.path)
    df = pd.DataFrame(tortuosity_metrics, columns =
                            ["fp_DM", "ir_DM", "fp_T_2", "ir_T_2", "fp_T_3", "ir_T_3", 
                                "fp_T_5", "ir_T_5", "fp_ASDC", "ir_ASDC", "fp_ICM", "ir_ICM",
                                "fp_TD", "ir_TD", "fp_SOAM", "ir_SOAM", "fp_SCC", "ir_SCC", "fp_ESCC", "ir_ESCC"])
    df.insert(0, "Curve Name", filenames)
    df.to_csv(args.output_file, index=False, float_format='%.8f')
    

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"Elapsed: {int(minutes)}m {seconds:.2f}s")