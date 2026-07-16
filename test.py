
import cv2
import math
import numpy as np


from Curve import Curve
from utils import geometry
import utils.load_and_write as lw


### path for the image and the curve coordinates
# im_path = "/Volumes/HOUSE MINI/IMAGENES/Fondus_Databases/SCALE_TORT_DB/im_arteries/18_18_art_fp.tif"
curve_path = 'data/curves_arteries/01_01_art_fp.txt'

### load the image and its corresponding curve coordinates, and create a Curve object
# img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
curve_coords_f = lw.load_pixelated_curve_from_txt_file(curve_path, delimiter=',')
curve_coords_o = lw.load_pixelated_curve_from_txt_file(curve_path.replace('_fp.txt', '_ir.txt'), delimiter=',')
curve_coords_f += 1  # add 1 to be MATLAB arteries_data.mat and veins_data.m compatible, (1 based indexing)
curve_coords_o += 1  # add 1 to be MATLAB arteries_data.mat and veins_data.m compatible, (1 based indexing)

pixel_curve_f = Curve(curve_coords_f[:, 0], curve_coords_f[:, 1])
pixel_curve_o = Curve(curve_coords_o[:, 0], curve_coords_o[:, 1])
# print(pixel_curve_f)
# print("Arc length:", pixel_curve_f.arclength())

Lf_c = pixel_curve_f.arclength()
Sf = math.ceil(len(pixel_curve_f) * 0.25)
Gf = 8 / Lf_c


Lf_c = pixel_curve_f.arclength()
Lo_c = pixel_curve_o.arclength()
Sf = math.ceil(len(pixel_curve_f) * 0.25)
So = math.ceil(len(pixel_curve_o) * 0.25)
Gf = 8 / Lf_c
Go = 8 / Lo_c


# my_smoother = SplineSmoother(smooth=0.02, k=5)
# smoothed_curve = pixel_curve.smooth(my_smoother, num_points=300)

smoothed_curve_f = pixel_curve_f.smooth("cubic_spline", smooth=Gf, num_points=So)
smoothed_curve_o = pixel_curve_o.smooth("cubic_spline", smooth=Go, num_points=So)

# param_curve = pixel_curve_f.resample(num_points=len(smoothed_curve), kind="linear")
Xf_scc, Yf_scc, _ = geometry.SCC_parametrization(smoothed_curve_f.x, smoothed_curve_f.y, n_points=len(smoothed_curve_f))
param_curve = Curve(Xf_scc, Yf_scc)



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


print("Tortuosity (DM): {:.5f}".format(T_dm))
print("Tortuosity (Total Curvature): {:.5f}".format(T_2))
print("Tortuosity (Tau_3): {:.5f}".format(T_3))
print("Tortuosity (Tau_5): {:.5f}".format(T_5))
print("Tortuosity (ASDC): {:.8f}".format(ASDC))
print("Tortuosity (ICM): {:.5f}".format(ICM))
print("Tortuosity (TD): {:.8f}".format(TD))
print("Tortuosity (SOAM): {:.5f}".format(SOAM))
print("Tortuosity (SCC): {:.5f}".format(T_scc))
print("Tortuosity (ESCC): {:.5f}".format(T_escc))

pixel_curve_f.plot(smoothed_curve_f, param_curve, labels=["original", "smoothed", "parametrized"], show_points=True)

# lw.display_curve_on_image(im_path, pixel_curve_f)

