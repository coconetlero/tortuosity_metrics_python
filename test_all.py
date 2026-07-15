
import cv2
import math
import numpy as np


from Curve import Curve
from utils import geometry
import utils.load_and_write as lw


output_file = "python_output.txt"

[af_curves, af_filenames] = lw.load_pixelated_curves_from_txt_files('data_II/curves_arteries/fp', delimiter=',')
[ao_curves, ao_filenames] = lw.load_pixelated_curves_from_txt_files('data_II/curves_arteries/ir', delimiter=',')

[vf_curves, vf_filenames] = lw.load_pixelated_curves_from_txt_files('data_II/curves_veins/fp', delimiter=',')
[vo_curves, vo_filenames] = lw.load_pixelated_curves_from_txt_files('data_II/curves_veins/ir', delimiter=',')

total_size = len(af_curves) + len(vf_curves)
tortuosity_measures = np.zeros((total_size, 20))

for af_curve, af_filename, ao_curve, ao_filename in zip(af_curves, af_filenames, ao_curves, ao_filenames):
    if af_filename.replace('_fp.txt', '') != ao_filename.replace('_ir.txt', ''):
            raise ValueError("Curves do not match: {} and {}".format(af_filename, ao_filename))

    # add 1 to loaded curve to be MATLAB compatible, (1 based indexing)
    af_curve += 1
    ao_curve += 1  
    
    af_pcurve = Curve(af_curve[:, 0], af_curve[:, 1])
    ao_pcurve = Curve(ao_curve[:, 0], ao_curve[:, 1])

    Lf_c = af_pcurve.arclength()
    Lo_c = ao_pcurve.arclength()
    Sf = math.ceil(len(af_pcurve) * 0.25)
    So = math.ceil(len(ao_pcurve) * 0.25)
    Gf = 8 / Lf_c
    Go = 8 / Lo_c

    smoothed_afcurve = af_pcurve.smooth("cubic_spline", smooth=Gf, num_points=So)
    smoothed_aocurve = ao_pcurve.smooth("cubic_spline", smooth=Go, num_points=So)

    Xfscc, Yfscc, _ = geometry.SCC_parametrization(
          smoothed_afcurve.x, smoothed_afcurve.y, n_points=len(smoothed_afcurve))
    Xoscc, Yoscc, _ = geometry.SCC_parametrization(
          smoothed_aocurve.x, smoothed_aocurve.y, n_points=len(smoothed_aocurve))
    param_fcurve = Curve(Xfscc, Yfscc)
    param_ocurve = Curve(Xoscc, Yoscc)


    T_dm_f = param_fcurve.tortuosity("DM")
    T_2_f = param_fcurve.tortuosity("total_curvature", smooth=1.0)
    T_3_f = param_fcurve.tortuosity("tau_3", smooth=1.0)
    T_5_f = param_fcurve.tortuosity("tau_5", smooth=1.0)
    ASDC_f = param_fcurve.tortuosity("ASDC", smooth=1.0)
    ICM_f = param_fcurve.tortuosity("ICM", smooth=1.0)
    TD_f = param_fcurve.tortuosity("TD", smooth=1.0)
    SOAM_f = param_fcurve.tortuosity("SOAM")
    T_scc_f = param_fcurve.tortuosity("SCC")
    T_escc_f = param_fcurve.tortuosity("ESCC")

    T_dm_o = param_ocurve.tortuosity("DM")
    T_2_o = param_ocurve.tortuosity("total_curvature", smooth=1.0)
    T_3_o = param_ocurve.tortuosity("tau_3", smooth=1.0)
    T_5_o = param_ocurve.tortuosity("tau_5", smooth=1.0)
    ASDC_o = param_ocurve.tortuosity("ASDC", smooth=1.0)
    ICM_o = param_ocurve.tortuosity("ICM", smooth=1.0)
    TD_o = param_ocurve.tortuosity("TD", smooth=1.0)
    SOAM_o = param_ocurve.tortuosity("SOAM")
    T_scc_o = param_ocurve.tortuosity("SCC")
    T_escc_o = param_ocurve.tortuosity("ESCC")

    tortuosity_measures[af_filenames.index(af_filename), :] = [
          T_dm_f, T_dm_o, T_2_f, T_2_o, T_3_f, T_3_o, T_5_f, T_5_o, ASDC_f, ASDC_o, 
          ICM_f, ICM_o, TD_f, TD_o, SOAM_f, SOAM_o, T_scc_f, T_scc_o, T_escc_f, T_escc_o]
    
    
print("Tortuosity measures computed for {} arteries.".format(len(af_curves)))
    

for vf_curve, vf_filename, vo_curve, vo_filename in zip(vf_curves, vf_filenames, vo_curves, vo_filenames):
    if vf_filename.replace('_fp.txt', '') != vo_filename.replace('_ir.txt', ''):
            raise ValueError("Curves do not match: {} and {}".format(vf_filename, vo_filename))

    # add 1 to loaded curve to be MATLAB compatible, (1 based indexing)
    vf_curve += 1
    vo_curve += 1  
    
    vf_pcurve = Curve(vf_curve[:, 0], vf_curve[:, 1])
    vo_pcurve = Curve(vo_curve[:, 0], vo_curve[:, 1])

    Lf_c = vf_pcurve.arclength()
    Lo_c = vo_pcurve.arclength()
    Sf = math.ceil(len(vf_pcurve) * 0.25)
    So = math.ceil(len(vo_pcurve) * 0.25)
    Gf = 8 / Lf_c
    Go = 8 / Lo_c

    smoothed_vfcurve = vf_pcurve.smooth("cubic_spline", smooth=Gf, num_points=So)
    smoothed_vocurve = vo_pcurve.smooth("cubic_spline", smooth=Go, num_points=So)

    Xfscc, Yfscc, _ = geometry.SCC_parametrization(
          smoothed_vfcurve.x, smoothed_vfcurve.y, n_points=len(smoothed_vfcurve))
    Xoscc, Yoscc, _ = geometry.SCC_parametrization(
          smoothed_vocurve.x, smoothed_vocurve.y, n_points=len(smoothed_vocurve))
    param_fcurve = Curve(Xfscc, Yfscc)
    param_ocurve = Curve(Xoscc, Yoscc)


    T_dm_f = param_fcurve.tortuosity("DM")
    T_2_f = param_fcurve.tortuosity("total_curvature", smooth=1.0)
    T_3_f = param_fcurve.tortuosity("tau_3", smooth=1.0)
    T_5_f = param_fcurve.tortuosity("tau_5", smooth=1.0)
    ASDC_f = param_fcurve.tortuosity("ASDC", smooth=1.0)
    ICM_f = param_fcurve.tortuosity("ICM", smooth=1.0)
    TD_f = param_fcurve.tortuosity("TD", smooth=1.0)
    SOAM_f = param_fcurve.tortuosity("SOAM")
    T_scc_f = param_fcurve.tortuosity("SCC")
    T_escc_f = param_fcurve.tortuosity("ESCC")

    T_dm_o = param_ocurve.tortuosity("DM")
    T_2_o = param_ocurve.tortuosity("total_curvature", smooth=1.0)
    T_3_o = param_ocurve.tortuosity("tau_3", smooth=1.0)
    T_5_o = param_ocurve.tortuosity("tau_5", smooth=1.0)
    ASDC_o = param_ocurve.tortuosity("ASDC", smooth=1.0)
    ICM_o = param_ocurve.tortuosity("ICM", smooth=1.0)
    TD_o = param_ocurve.tortuosity("TD", smooth=1.0)
    SOAM_o = param_ocurve.tortuosity("SOAM")
    T_scc_o = param_ocurve.tortuosity("SCC")
    T_escc_o = param_ocurve.tortuosity("ESCC")

    tortuosity_measures[len(af_curves) + vf_filenames.index(vf_filename), :] = [
          T_dm_f, T_dm_o, T_2_f, T_2_o, T_3_f, T_3_o, T_5_f, T_5_o, ASDC_f, ASDC_o, 
          ICM_f, ICM_o, TD_f, TD_o, SOAM_f, SOAM_o, T_scc_f, T_scc_o, T_escc_f, T_escc_o]

print("Tortuosity measures computed for {} veins.".format(len(vf_curves)))

print("Saving tortuosity measures to {}...".format(output_file))
np.savetxt(output_file, tortuosity_measures, delimiter="\t", fmt="%.8f")
# np.savetxt(output_file, tortuosity_measures, delimiter="\t", fmt="%.6e")