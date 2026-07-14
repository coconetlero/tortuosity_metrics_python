
import cv2
import math
import numpy as np


from Curve import Curve
from utils import geometry
import utils.load_and_write as lw



# im_path = "/Volumes/HOUSE MINI/IMAGENES/Fondus_Databases/SCALE_TORT_DB/im_arteries/18_18_art_fp.tif"
curve_path = 'data/curves_arteries/18_18_art_fp.txt'

# img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
curve_coords = lw.load_pixelated_curve_from_txt_file(curve_path, delimiter=',')
curve_coords += 1  # add 1 to be MATLAB arteries_data.mat and veins_data.m compatible, (1 based indexing)

pixel_curve = Curve(curve_coords[:, 0], curve_coords[:, 1])
print(pixel_curve)
print("Arc length:", pixel_curve.arclength())

Lf_c = pixel_curve.arclength()
Sf = math.ceil(len(pixel_curve) * 0.25)
Gf = 8 / Lf_c


# my_smoother = SplineSmoother(smooth=0.02, k=5)
# smoothed_curve = pixel_curve.smooth(my_smoother, num_points=300)

smoothed_curve = pixel_curve.smooth("cubic_spline", smooth=Gf, num_points=Sf)

# param_curve = pixel_curve.resample(num_points=len(smoothed_curve), kind="linear")
Xscc, Yscc, _ = geometry.SCC_parametrization(smoothed_curve.x, smoothed_curve.y, n_points=len(smoothed_curve))
param_curve = Curve(Xscc, Yscc)



# T_dm = param_curve.tortuosity("DM")
# T_2 = param_curve.tortuosity("total_curvature", smooth=1.0)
T_3 = param_curve.tortuosity("tau_3", smooth=1.0)
T_5 = param_curve.tortuosity("tau_5", smooth=1.0)
# ASDC = param_curve.tortuosity("ASDC", smooth=1.0)
# ICM = param_curve.tortuosity("ICM", smooth=1.0)
# TD = param_curve.tortuosity("TD", smooth=1.0)
# SOAM = param_curve.tortuosity("SOAM")
# T_scc = param_curve.tortuosity("SCC")
# T_escc = param_curve.tortuosity("ESCC")


# print("Tortuosity (DM): {:.4f}".format(T_dm))
# print("Tortuosity (Total Curvature): {:.4f}".format(T_2))
print("Tortuosity (Tau_3): {:.4f}".format(T_3))
print("Tortuosity (Tau_5): {:.4f}".format(T_5))
# print("Tortuosity (ASDC): {:.4f}".format(ASDC))
# print("Tortuosity (ICM): {:.4f}".format(ICM))
# print("Tortuosity (TD): {:.4f}".format(TD))
# print("Tortuosity (SOAM): {:.4f}".format(SOAM))
# print("Tortuosity (SCC): {:.4f}".format(T_scc))
# print("Tortuosity (ESCC): {:.4f}".format(T_escc))

pixel_curve.plot(smoothed_curve, param_curve, labels=["original", "smoothed", "parametrized"], show_points=True)

# lw.display_curve_on_image(im_path, pixel_curve)

