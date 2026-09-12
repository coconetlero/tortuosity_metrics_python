import os
import re
import time
import cv2
import math
import argparse
import pandas as pd
import numpy as np


from Curve import Curve
from utils import geometry
import utils.load_and_write as lw


def get_filenames(folder_path):
    """
    Get all filenames from a folder containing txt files
    """
    filenames = []
    filtered_filenames = [item for item in os.listdir(folder_path) if not item.startswith('.')]
    filtered_filenames = sorted(filtered_filenames)
    for filename in filtered_filenames:
        filenames.append(filename)
    
    return filenames



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



def process_data(images_path, curves_filenames, curves, num_of_points):
    
    filenames = []
    tortuosity_metrics = []
    for filename, curve in zip(curves_filenames, curves):
        raster_curve = Curve(curve[:, 0], curve[:, 1])

        # if filename.replace('_fp.txt', '') != filename.replace('_ir.txt', ''):
        #         raise ValueError("Curves do not match: {} and {}".format(fp_filename, ir_filename))

        raster_curve = Curve(curve[:, 0], curve[:, 1])
        im_filename = os.path.join(images_path, filename.replace('.txt', '.tif'))
        print(filename)
        lw.display_curve_on_image(im_filename, raster_curve.points)
        L_c = raster_curve.arclength()

        smoothed_curve = raster_curve.smooth("scale_cubic_spline", num_points=num_of_points)  
        param_curve = smoothed_curve.parametrize("scc", num_points=len(smoothed_curve)) 

        tm = compute_tortuosities(param_curve)
        tortuosity_metrics.append(tm)
        filenames.append(filename.replace('.txt', ''))
        

        # filenames[i] = re.sub(r'_(fp|ir)\..*$', '', fp_filename)
        # T = [TC_fp, TC_ir, T_scc_fp, T_scc_ir, T_escc_fp, T_escc_ir]
        # tortuosity_metrics[i, :] = T
        # # print("{} - parm points: {},\tGf: {:.5f}".format(vfp_filename, len(param_curve_fp), Gf))        
        # print(f"{re.sub(r'_(fp|ir)\..*$', '', fp_filename)}: " + ", ".join("{:.6f}".format(x) for x in T))


    return tortuosity_metrics, filenames

    

def main():
    # parser = argparse.ArgumentParser(description='Compute tortuosity metrics for a given curve')
    # parser.add_argument('path', help='Path to folder containing the curve txt files')
    # parser.add_argument('output_file', help='Path to the output file')
    # args = parser.parse_args()



    # tortuosity_metrics, filenames = process_data(args.path)
    images_path = "/Users/zianfanti/IIMAS/Fondus_Images_Databases/RET-TORT/Reduced Arteries Iso"
    curves_path = "data/RET_TORT/arteries"
    
    """ ---------- Data loading ---------- """    
    curves_filenames = get_filenames(curves_path)
    curves = lw.load_curves_from_filenames(curves_path, curves_filenames)

    """ ---------- End Data loading ---------- """



    tortuosity_metrics, filenames = process_data(images_path, curves_filenames, curves, 200)


    df = pd.DataFrame(tortuosity_metrics, columns = 
                      ["DM", "T_2", "T_3", "T_5", "ASDC", "ICM", "TD", "SOAM", "SCC", "ESCC"])
    df.insert(0, "Curve Name", filenames)
    df.to_csv(args.output_file, index=False, float_format='%.8f')


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"Elapsed: {int(minutes)}m {seconds:.2f}s")