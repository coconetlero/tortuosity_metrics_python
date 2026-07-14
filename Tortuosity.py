"""
tortuosity.py

Implements the Strategy pattern for curve tortuosity measurements.

Each TortuosityMeasure subclass encapsulates one way of quantifying tortuosity. 
New measures can be added by subclassing `TortuosityMeasure` without modifying any existing code.

Usage:
    measure = ArcChordTortuosity()
    value = measure.compute(x, y)

    or via the registry / Curve.tortuosity(method="arc_chord")
"""

import numpy as np
from abc import ABC, abstractmethod

import utils.geometry as geometry
from utils.Derivatives import CurveDerivatives


class _ArrayCurve:
    """
    Minimal duck-typed stand-in for a Curve, exposing just enough
    (.x, .y, .arclength(), len()) for CurveDerivatives to operate on
    raw x, y arrays without constructing a full Curve instance.
    """
 
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    def arclength(self):
        return geometry.arclength(self.x, self.y)

    def __len__(self):
        return len(self.x)
    


class TortuosityMeasure(ABC):
    """
    Abstract base class for all tortuosity measurement strategies.

    Subclasses must implement `compute`, which takes the curve's x, y
    coordinate arrays and returns a single scalar tortuosity value.
    """

    @abstractmethod
    def compute(self, x, y):
        """
        Computes a tortuosity value for the given curve.

        Parameters:
            x, y : np.ndarray
                Coordinates of the curve.

        Returns:
            float
                The computed tortuosity value. Interpretation depends
                on the specific measure (see each subclass docstring).
        """
        raise NotImplementedError
    



class Distance_Metric_Tortuosity(TortuosityMeasure):
    """
    Classic arc-chord ratio (a.k.a. Distance Metric, DM):

        tortuosity = arc_length / chord_length

    where chord_length is the straight-line distance between the
    curve's start and end points.

    Returns 1.0 for a perfectly straight curve; values increase as the
    curve becomes more convoluted. Undefined (returns np.inf) if start
    and end points coincide.
    """

    def compute(self, x, y):
        arc_length = geometry.arclength(x, y)
        chord = geometry.chord_length(x, y)
 
        if np.isclose(chord, 0):
            return np.inf
 
        return arc_length / chord
    

    

class MeanCurvatureTortuosity(TortuosityMeasure):
    """
    Mean absolute curvature along the curve, 
 
        curvature(s) = |kappa(s)|
        tortuosity   = mean(curvature(s))
 
    Higher values indicate a curve that bends more sharply on average.
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact.
            interpolation. Default 0.99 lightly regularizes to reduce
            noise sensitivity in the 2nd derivative while staying close
            to the original curve. 
    """
 
    def __init__(self, smooth=0.99):
        self.smooth = smooth
 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        deriv = CurveDerivatives(_ArrayCurve(x, y))
        _, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
 
        return float(np.mean(np.abs(kappa)))
    



class TotalCurvatureTortuosity(TortuosityMeasure):
    """
    Total Curvature (Smedby et al., 1993): the integral of absolute curvature along 
    the curve's arc length, also known as Tau_2 by (William E. Hart et al., 1998):
    
        TC = integral of |kappa(s)| ds
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact.
            interpolation. Default 0.99 lightly regularizes to reduce
            noise sensitivity in the 2nd derivative while staying close
            to the original curve. 
    """
 
    def __init__(self, smooth=0.99):
        self.smooth = smooth

 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0

        curve_parametrization = geometry.cumulative_arclength(x, y)
        deriv = CurveDerivatives(_ArrayCurve(x, y), t=curve_parametrization)
        t, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
        total_curvature = np.trapezoid(np.abs(kappa), t)
 
        return float(total_curvature)



class Tau_3_Tortuosity(TortuosityMeasure):
    """
    Total Squared Curvature (Tau_3) (William E. Hart et al., 1998):
    
        Tau_3 = TSC = integral of kappa(s)^2 ds
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact.
            interpolation. Default 0.99 lightly regularizes to reduce
            noise sensitivity in the 2nd derivative while staying close
            to the original curve. 
    """
 
    def __init__(self, smooth=0.99):
        self.smooth = smooth

 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        curve_parametrization = geometry.cumulative_arclength(x, y)
        deriv = CurveDerivatives(_ArrayCurve(x, y), t=curve_parametrization)
        t, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
        Tau_3 = np.trapezoid(kappa ** 2, t)
 
        return float(Tau_3)



class Tau_5_Tortuosity(TortuosityMeasure):
    """
    Tau_5 Tortuosity (William E. Hart et al., 1998):
    
        Tau_5 = Tau_3 / arc_length
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact.
            interpolation. Default 0.99 lightly regularizes to reduce
            noise sensitivity in the 2nd derivative while staying close
            to the original curve. 
    """
 
    def __init__(self, smooth=0.99):
        self.smooth = smooth

 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        curve_parametrization = geometry.cumulative_arclength(x, y)
        deriv = CurveDerivatives(_ArrayCurve(x, y), t=curve_parametrization)
        t, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
        Tau_3 = np.trapezoid(kappa ** 2, t)
        L_c = geometry.arclength(x, y)
        Tau_5 = Tau_3 / L_c

        return float(Tau_5)




class AverageSquaredDerivativeCurvatureTortuosity(TortuosityMeasure):
    """
    Average Squared-Derivative-Curvature (ASDC), based on Patasius et
    al. (used in retinal vessel tortuosity literature):
 
        ASDC = (1 / L) * integral( (d(kappa)/ds)^2 ) ds


    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit for kappa(s). 0 = least-squares line fit, 1 =
            exact interpolation. Default 0.99. Requires:
            pip install csaps.

    """

    def __init__(self, smooth=0.99):
        self.smooth = smooth


    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        curve = CurveDerivatives(_ArrayCurve(x, y))
        t, kappa = curve.curvature(method="csaps", smooth=self.smooth)
 
        trim = int(round(0.1 * len(t)))
        if trim > 0 and len(t) - 2 * trim >= 3:
            t = t[trim: len(t) - trim]
            kappa = kappa[trim: len(kappa) - trim]
 
        if len(t) < 3:
            return 0.0
 
        dkappa_ds = np.gradient(kappa, t)
        integral = np.trapezoid(dkappa_ds**2, t)
 
        L = t[-1] - t[0]
        if np.isclose(L, 0):
            return 0.0
 
        return float(integral / L)




class InflectionCountTortuosity(TortuosityMeasure):
    """
    Inflection Count Metric (ICM), based on Bullitt et al. (2003):
 
        ICM = (arc_length / chord_length) * (n_inflections + 1)
 
    where n_inflections is the number of sign changes in curvature
    (i.e. the number of times the curve switches from bending one way
    to bending the other), computed using exact analytical derivatives
    from a csaps cubic smoothing spline fit (see derivatives.py /
    CurveDerivatives / CsapsDerivative). Captures both overall
    elongation (via the arc-chord ratio) and the number of directional
    reversals.
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact
            interpolation. Default 0.99 lightly regularizes to avoid
            spurious inflections from point-to-point noise. Requires:
            pip install csaps.
    """
 
    def __init__(self, smooth=0.99):
        self.smooth = smooth
 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        deriv = CurveDerivatives(_ArrayCurve(x, y))
        _, signed_curvature = deriv.curvature(method="csaps", smooth=self.smooth)
 
        signs = np.sign(signed_curvature)
        signs = signs[signs != 0]  # ignore exact zeros
        n_inflections = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
 
        arc_chord = Distance_Metric_Tortuosity().compute(x, y)
        return arc_chord * (n_inflections + 1)




class SOAMTortuosity(TortuosityMeasure):
    """
    Sum of Angles Metric (SOAM), based on Bullitt et al. (2003).

    Computes the discrete turning angle at each interior point and
    accumulates their squared magnitude, normalized by arc length:

        SOAM = sqrt( sum(angle_i^2) / arc_length )

    Higher values indicate more frequent/sharper directional changes
    per unit length. Sensitive to high-frequency noise, so consider
    smoothing the curve first (see smoothers.py / Curve.smooth()).
    """

    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0

        angles = np.zeros(n - 2)
        for i in range(1, n - 1):
            v1 = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
            v2 = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if np.isclose(norm1, 0) or np.isclose(norm2, 0):
                angles[i - 1] = 0.0
                continue

            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            angles[i - 1] = np.arccos(cos_angle)

        dx = np.diff(x)
        dy = np.diff(y)
        arc_length = np.sum(np.sqrt(dx**2 + dy**2))

        if np.isclose(arc_length, 0):
            return 0.0

        return np.sqrt(np.sum(angles**2) / arc_length)
    



class TortuosityDensityTortuosity(TortuosityMeasure):
    """
    Tortuosity Density (TD), based on T. D. Nafia et al.
 
    The curve is split into segments at each inflection point (a sign
    change in curvature — i.e. where the curve switches from bending
    one way to bending the other). For each segment, the "excess
    length" relative to a straight line between its endpoints is
    computed, and these are summed and normalized by the total chord
    length of the whole curve:
 
        TD = (1 / Lx) * sum_{i=1}^{N} (Lc_i / Lx_i - 1)
 
    where Lc_i and Lx_i are the arc length and chord length of the
    i-th segment (N segments total, split at N-1 inflection points),
    and Lx is the chord length of the entire curve.

    Parameters:
    smooth : float
        Smoothing parameter in [0, 1], passed to the underlying
        csaps fit used to locate inflection points. 0 =
        least-squares line fit, 1 = exact interpolation. Default
        0.99 lightly regularizes to avoid spurious inflections
        from point-to-point noise. Requires: pip install csaps.
    """


    def __init__(self, smooth=0.99):
        self.smooth = smooth
 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        L_x = geometry.chord_length(x, y)
        if np.isclose(L_x, 0):
            return np.inf
 
        deriv = CurveDerivatives(_ArrayCurve(x, y))
        _, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
 
        # Find indices where curvature changes sign 
        signs = np.sign(kappa)
        boundaries = [0]
        prev_sign = None
        for i in range(n):
            s = signs[i]
            if s == 0:
                continue
            if prev_sign is not None and s != prev_sign:
                boundaries.append(i)
            prev_sign = s
        boundaries.append(n - 1)
        boundaries = sorted(set(boundaries))
 
        segment_sum = 0.0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            seg_x = x[start:end + 1]
            seg_y = y[start:end + 1]
 
            Lc_i = geometry.arclength(seg_x, seg_y)
            Lx_i = geometry.chord_length(seg_x, seg_y)
 
            if np.isclose(Lx_i, 0):
                continue  # degenerate (zero-length) segment, skip
 
            segment_sum += (Lc_i / Lx_i - 1)
 
        return float(segment_sum / L_x)





class SCC_Tortuosity(TortuosityMeasure):
    """
    
    """

    def scc(self, x, y):
        diff_x = np.diff(x)
        diff_y = np.diff(y)
                
        Theta = np.degrees(np.arctan2(diff_y, diff_x))
        alpha = np.diff(Theta)

        C = np.where((alpha >= 180) | (alpha <= -180))[0]
        for k in C: 
            alpha[k] = np.mod(alpha[k], np.sign(alpha[k]) * -360)
        
        alpha = alpha/180
        return alpha

    def compute(self, X, Y):
        alpha = self.scc(X, Y)
        SCC = np.sum(np.absolute(alpha))        
 
        return SCC
    


class ESCC_Tortuosity(TortuosityMeasure):
    """
    
    """

    def escc(self, X, Y):
        n = len(X)
        Xe = []
        Ye = []
    
        L_x = np.mean(geometry.segment_lengths(X, Y))
    
        vx = X[1] - X[0]
        vy = Y[1] - Y[0]
        M = np.degrees(np.arctan2(vy, vx))
    
        x_0 = X[0]
        y_0 = Y[0]
            
        x_1 = y_1 = None
        Xl = Yl = None
        Xp = Yp = None
    
        idx = 0
        while (idx >= 0) and (idx < n - 1):
            # heart shaped (box) pattern construction 
            Xe.append(x_0)
            Ye.append(y_0)

            alpha = np.radians(M)
            theta = np.linspace(alpha, alpha + np.pi, 180)

            Xp = x_0 + (L_x * (1 - ((theta - alpha) / np.pi))) * np.cos(theta)
            Yp = y_0 + (L_x * (1 - ((theta - alpha) / np.pi))) * np.sin(theta)

            theta = np.linspace(alpha, alpha - np.pi, 180)
            Xn = x_0 + (L_x * (1 + ((theta - alpha) / np.pi))) * np.cos(theta)
            Yn = y_0 + (L_x * (1 + ((theta - alpha) / np.pi))) * np.sin(theta)

            Xt = np.concatenate([Xp[::-3], Xn])
            Xt = Xt[1:-3]
            Yt = np.concatenate([Yp[::-3], Yn])
            Yt = Yt[1:-3]

            intersection = False
            n_idx = 1
            m_idx = 0

            # find intersection between the heart-shaped pattern and the original curve
            while not intersection:
                if idx < 0 or idx >= n - 1:
                    break
 
                Xl = (X[idx], X[idx + 1])
                Yl = (Y[idx], Y[idx + 1])
                
                xi, yi = geometry.segment_polyline_intersections(
                    (Xl[0], Yl[0]), (Xl[1], Yl[1]), Xt, Yt
                )
 
                if len(xi) > 0:
                    if len(xi) > 1:
                        dists = np.sqrt(
                            (x_0 - np.array(xi)) ** 2 + (y_0 - np.array(yi)) ** 2
                        )
                        j = int(np.argmax(dists))
                        x_1 = xi[j]
                        y_1 = yi[j]
                    else:
                        x_1 = xi[0]
                        y_1 = yi[0]
                    intersection = True
                else:
                    if n_idx != m_idx:
                        if (idx + n_idx) < n - 1:
                            m_idx += 1
                            idx = idx + m_idx
                            n_idx += 1
                            m_idx = n_idx
                        else:
                            d1 = np.hypot(X[idx] - X[idx + 1], Y[idx] - Y[idx + 1])
                            d2 = np.hypot(Xp[0] - Xp[-1], Yp[0] - Yp[-1])
                            if d1 < d2:
                                break
                            else:
                                idx = idx - n_idx
                    else:
                        idx = idx - n_idx
                        n_idx += 1
 
            vx = x_1 - x_0
            vy = y_1 - y_0
            M = np.degrees(np.arctan2(vy, vx))
 
            idx += 1
 
            x_0 = x_1
            y_0 = y_1
 
        Xe.append(X[-1])
        Ye.append(Y[-1])

    
        return np.array(Xe), np.array(Ye)        


    def scc(self, X, Y):
        diff_x = np.diff(X)
        diff_y = np.diff(Y)

        Theta = np.degrees(np.arctan2(diff_y, diff_x))
        alpha = np.diff(Theta)

        C = np.where((alpha >= 180) | (alpha <= -180))[0]
        for k in C: 
            alpha[k] = np.mod(alpha[k], np.sign(alpha[k]) * -360)
        
        alpha = alpha/180
        return alpha



    def compute(self, X, Y):
        [Xe,Ye] = self.escc(X,Y);
        alpha = self.scc(Xe,Ye);
        ESCC = np.sum(np.absolute(alpha))        
 
        return ESCC
    
    


# ----------------------------------------------------------------------
# Registry: maps a string method name -> TortuosityMeasure class
# ----------------------------------------------------------------------

TORTUOSITY_REGISTRY = {
    "DM": Distance_Metric_Tortuosity,
    "mean_curvature": MeanCurvatureTortuosity,
    "total_curvature": TotalCurvatureTortuosity,
    "tau_3": Tau_3_Tortuosity,
    "tau_5": Tau_5_Tortuosity,
    "ASDC": AverageSquaredDerivativeCurvatureTortuosity,
    "ICM": InflectionCountTortuosity,
    "SOAM": SOAMTortuosity,    
    "TD": TortuosityDensityTortuosity,
    "SCC": SCC_Tortuosity,
    "ESCC": ESCC_Tortuosity,
}


def get_tortuosity_measure(method, **kwargs):
    """
    Factory function: builds a TortuosityMeasure instance from a
    registry key.

    Parameters:
        method : str
            One of the keys in TORTUOSITY_REGISTRY
            (e.g. 'arc_chord', 'soam', 'mean_curvature', 'icm').
        **kwargs :
            Passed to the measure's constructor.

    Returns:
        TortuosityMeasure instance
    """
    if method not in TORTUOSITY_REGISTRY:
        available = ", ".join(TORTUOSITY_REGISTRY.keys())
        raise ValueError(
            f"Unknown tortuosity method '{method}'. Available: {available}"
        )
    return TORTUOSITY_REGISTRY[method](**kwargs)
