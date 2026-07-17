# Curve Toolkit

Tools for representing 2D curves and computing smoothing, derivatives, and
tortuosity measures on them.

## Requirements

- Python 3.12 (developed and tested with 3.12.11)

## Setup

```bash
git clone https://github.com/coconetlero/tortuosity_metrics_python.git
cd tortuosity_metrics_python

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Verify the install

```bash
python3 -c "from curve import Curve; c = Curve([0, 1, 2], [0, 1, 4]); print(c)"
```

Expected output:

```
Curve(n_points=3, length=4.5765)
```

## Run
The metrics test file:

python test.py

## Modules

- `curve.py` — `Curve` class: 2D open curve, smoothing, resampling, plotting
- `smoothers.py` — smoothing strategies (spline, moving average, Savitzky-Golay, Gaussian)
- `derivatives.py` — `CurveDerivatives`: finite differences, spline, and csaps derivative methods
- `tortuosity.py` — tortuosity measures (arc-chord, SOAM, curvature-based, ICM, TD, ASDC)
- `geometry.py` — shared arc-length / chord-length utilities


## Data 

Data are available in the data and data_II folders
all the arteries curves including Fundus and OCT cameras
- data/curves_arteries/  

all the veins curves including Fundus and OCT cameras
- data/curves_veins/

arteries from Fundus images
data_II/curves_arteries/fp
arteries from OCT images
data_II/curves_arteries/ir

veins from Fundus images
data_II/curves_veins/fp
veins from OCT images
data_II/curves_veins/ir

For the corresponding images you can download the SCALE-TORT Database in the Leipzig Health Atlas

https://www.health-atlas.de/projects/64
