import argparse
import math
import numpy as np
import pandas as pd
import sys

import utils.load_and_write as lw
from Curve import Curve

import time


def process_folder(folder_path, output_file):
    raw_curves, filenames = lw.load_curves_from_Folder(folder_path)

    tortuosity_measures = []
    for raw_curve in raw_curves:
        # raw_curve += 1  # add 1 to be MATLAB compatible, (1 based indexing)
        
        raster_curve = Curve(raw_curve[:, 0], raw_curve[:, 1])

        # --- proposed heuristics for smoothing and resampling ---
        # --- you can change with yours ---
        L_c = raster_curve.arclength()
        num_of_points = math.ceil(len(raster_curve) * 0.25)
        smoothness = 8 / L_c

         # --- smooth and resample the curve
        smoothed_curve = raster_curve.smooth("cubic_spline", smooth=smoothness, num_points=num_of_points)
        param_curve = smoothed_curve.parametrize("scc", num_points=num_of_points)

        # --- compute tortuosity metrics ---
        
        T_dm = param_curve.tortuosity("DM")
        T_2 = param_curve.tortuosity("total_curvature", smooth=1.0)
        T_3 = param_curve.tortuosity("tau_3", smooth=1.0)
        T_5 = param_curve.tortuosity("tau_5", smooth=1.0)
        ASDC = param_curve.tortuosity("ASDC", smooth=1.0)
        ICM = param_curve.tortuosity("ICM", smooth=1.0)
        TD = param_curve.tortuosity("TD", smooth=1.0)
        SOAM = param_curve.tortuosity("SOAM")
        T_scc = param_curve.tortuosity("SCC")
        T_escc = param_curve.tortuosity("ESCC")

        tortuosity_measures.append([T_dm, T_2, T_3, T_5, ASDC, ICM, TD, SOAM, T_scc, T_escc])

    df = pd.DataFrame(tortuosity_measures, columns =
                      ["DM", "T_2", "T_3", "T_5", "ASDC", "ICM", "TD", "SOAM", "SCC", "ESCC"])
    df.insert(0, "Curve Name", filenames)
    df.to_csv(output_file, index=False, float_format='%.6f')



def main():
    parser = argparse.ArgumentParser(description='Compute tortuosity metrics for a given curve')
    parser.add_argument('path', help='Path to folder containing the curve txt files')
    parser.add_argument('output_file', help='Path to the output file')
    args = parser.parse_args()

    try:
        process_folder(args.path, args.output_file)
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    
    print(f"Execution Time: {elapsed:.6f} segundos")