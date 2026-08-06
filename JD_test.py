import os
import re
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


def load_curves_from_filenames(folder_path, filenames):
    """
    Load all curves from a list of filenames
    """
    curves = []
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        curve = lw.load_curve_from_txt_file(file_path)
        curves.append(curve)
    
    return curves


if __name__ == "__main__":

    """ ---------- Data loading ---------- """    

    veins_path = 'data/original_data/veins_data_coords'    
    veins_filenames = get_filenames(veins_path)
    
    vfp_filenames = [f for f in veins_filenames if re.search(r'_fp(?:_|\.|$)', f)]
    vir_filenames = [f for f in veins_filenames if re.search(r'_ir(?:_|\.|$)', f)]

    vfp_curves = load_curves_from_filenames(veins_path, vfp_filenames)
    vir_curves = load_curves_from_filenames(veins_path, vir_filenames)

    """ ---------- End Data loading ---------- """

    filenames = [""] * len(vfp_curves)
    tortuosity_measures = np.zeros((len(vfp_curves), 4))
    for i in range(len(vfp_curves)):
        vfp_filename = vfp_filenames[i]
        vir_filename = vir_filenames[i]

        if vfp_filename.replace('_fp.txt', '') != vir_filename.replace('_ir.txt', ''):
                raise ValueError("Curves do not match: {} and {}".format(vfp_filename, vir_filename))

        pixel_curve_fp = Curve(vfp_curves[i][:, 0], vfp_curves[i][:, 1])
        pixel_curve_ir = Curve(vir_curves[i][:, 0], vir_curves[i][:, 1])

        Lf_c = pixel_curve_fp.arclength()
        Lo_c = pixel_curve_ir.arclength()
        Sf = math.ceil(len(pixel_curve_fp) * 0.25)
        So = math.ceil(len(pixel_curve_ir) * 0.25)
        Gf = 8 / Lf_c
        Go = 8 / Lo_c

        smoothed_curve_fp = pixel_curve_fp.smooth("cubic_spline", smooth=Gf, num_points=Sf)
        smoothed_curve_ir = pixel_curve_ir.smooth("cubic_spline", smooth=Go, num_points=So)

        Xfp_scc, Yfp_scc, _ = geometry.SCC_parametrization(smoothed_curve_fp.x, smoothed_curve_fp.y, n_points=len(smoothed_curve_fp))
        Xir_scc, Yir_scc, _ = geometry.SCC_parametrization(smoothed_curve_ir.x, smoothed_curve_ir.y, n_points=len(smoothed_curve_ir))

        param_curve_fp = Curve(Xfp_scc, Yfp_scc)
        param_curve_ir = Curve(Xir_scc, Yir_scc)

        T_scc_fp = param_curve_fp.tortuosity("SCC")
        T_escc_fp = param_curve_fp.tortuosity("ESCC")

        T_scc_ir = param_curve_ir.tortuosity("SCC")
        T_escc_ir = param_curve_ir.tortuosity("ESCC")

        filenames[i] = re.sub(r'_(fp|ir)\..*$', '', vfp_filename)
        tortuosity_measures[i, :] = [T_scc_fp, T_escc_fp, T_scc_ir, T_escc_ir]
        # print("{} - parm points: {},\tGf: {:.5f}".format(vfp_filename, len(param_curve_fp), Gf))


    df = pd.DataFrame(tortuosity_measures, columns=["SCC FP", "ESCC FP", "SCC IR", "ESCC IR"])
    df.insert(0, "Curve Name", filenames)
    df.to_csv("tort_results_(JD).csv", index=False, float_format='%.8f')