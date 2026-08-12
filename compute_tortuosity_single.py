import math
import argparse
import sys
import numpy as np


from Curve import Curve
import utils.load_and_write as lw


def process_curve(curve_path):
    raw_curve = lw.load_curve_from_txt_file(curve_path)
    # curve_coords += 1  # add 1 to be MATLAB compatible, (1 based indexing)

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
    tortuosity_metrics = {
        "T_dm": param_curve.tortuosity("DM"),
        "T_2": param_curve.tortuosity("total_curvature", smooth=1.0),
        "T_3": param_curve.tortuosity("tau_3", smooth=1.0),
        "T_5": param_curve.tortuosity("tau_5", smooth=1.0),
        "ASDC": param_curve.tortuosity("ASDC", smooth=1.0),
        "ICM": param_curve.tortuosity("ICM", smooth=1.0),
        "TD": param_curve.tortuosity("TD", smooth=1.0),
        "SOAM": param_curve.tortuosity("SOAM"),
        "T_scc": param_curve.tortuosity("SCC")
    }

    print(f"Tortuosity metrics for {curve_path}:")
    for metric, value in tortuosity_metrics.items():
        print("{}: {:.6f}".format(metric, value))


def main():
    parser = argparse.ArgumentParser(description='Compute tortuosity metrics for a given curve')
    parser.add_argument('path', help='Path to the txt file')
    args = parser.parse_args()
    
    try:        
        process_curve(args.path)
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()