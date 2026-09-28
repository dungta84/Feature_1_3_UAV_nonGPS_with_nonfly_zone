# Visibility-based shape vector for GPS-denied UAV localization

Code for Examples 4.2 and 4.3 of the manuscript "Lebesgue Integral-based Shape
Vector Descriptors with Applications to GPS-Denied UAV Localization"
(submitted to Pattern Recognition).

At a polygon vertex x* the shape vector W(x*) holds the building area in each
cell of a disk of radius R around x*, split into n rings and 2^k sectors
(f = 1). Aerial images are matched against a reference map by comparing these
vectors; the reference buildings that pass the test are the candidates handed
to a fine matching step.

## Quick start (Windows)

1. `pip install -r requirements.txt` (Python 3.10 or newer, shapely 2.x)
2. Double-click `run_examples.bat`. It recomputes both examples and writes
   `results_examples/`; the log is `results_examples/run_log.txt`.
3. Double-click `plot_figures.bat`. It draws the reference maps, the descriptor
   figure and a localization map for every test image. The figures need the
   feature files of step 2 (they are not stored in the repository); if they are
   missing, `plot_figures.bat` runs `run_examples.py` first.

On other systems run the same steps with `python run_examples.py`,
`python plot_reference_maps.py`, `python plot_descriptor.py`,
`python plot_localization.py`.

## Descriptor vertices: extreme points of the COCH

Every building polygon is the connected orthogonal convex hull (COCH) of its
contour, computed by the modified Graham scan in `OGraham.py` (called from
`OrthoHullLib.py`). The COCH is an (x, y)-polygon whose convex vertices are its
extreme points, which belong to the contour, and whose reflex vertices are the
corners of the staircases between them (An, Huyen and Le, Appl. Math. Comput.
397 (2021) 125889). The shape vector is used at the extreme vertices only. The
feature CSV files keep every vertex, with a column `vertex_type` (`convex`,
`reflex`, `flat`), so that the polygon can be rebuilt;
`run_matching_all_v3.py` uses the `convex` rows
(`VERTEX_MODE = "coch_extreme"`). `python run_examples.py --vertices all` gives
the earlier behaviour (every vertex).

## What `run_examples.py` does

For each example (`ex4_2` without, `ex4_3` with the no-fly zone):

1. Extracts the buildings of `references/refmap.png` (threshold 18, minimum
   area 2850 px; 12 buildings) and computes R = 0.95 x the smallest distance
   from a polygon vertex to another polygon (the no-fly zone is included in
   `ex4_3`). Both examples give R = 5.83 m.
2. Computes the reference shape vectors for n, k in {3, 4, 5}.
3. Cuts one 330 x 204 px test image around each of the 12 buildings. In
   `ex4_3` the no-fly zone is an obstacle: images that meet it are dropped,
   and its area is not added to the shape vector.
4. Extracts the buildings of every test image with a lower threshold (16,
   minimum area 1250 px), as a second sensor would, and computes their shape
   vectors.
5. Runs `run_matching_all_v3.py`: L1 distance between shape vectors, only
   vertices whose disk lies inside the image, a reference building is a
   candidate when at least half of the vertices of the query have a vertex of
   it within eps. One eps for all (n, k). The ground truth comes from the crop
   offsets and is used only for evaluation.

The feature computation (`features/fast_features.py`) is a vectorized version
of the original loop in `features/core.py` (switch with `USE_FAST_FEATURES`);
both give the same numbers.

## Results (as in the paper)

Mean number of candidates, recall of the true building and top-1 accuracy:

HSV threshold, descriptor at the COCH extreme vertices, R = 5.83 m:

| Example | queries | (3,3) | (5,5) | recall | top-1 |
|---|---|---|---|---|---|
| 4.2, no no-fly zone | 12 | 3.42 | 2.83 | 100% | 11/12 |
| 4.3, no-fly zone | 9 | 3.89 | 3.22 | 100% | 8/9 |

Full tables: `results_examples/<example>/matching/summary_nk.csv`,
`per_query.csv`, `per_test.csv`. The position error of a correct match is
zero because the test images are cut from the reference map; it measures the
matching, not the accuracy of a real flight.

## Files

| Path | Content |
|---|---|
| `OGraham.py`, `OrthoHullLib.py` | connected orthogonal convex hull (modified Graham scan) |
| `features/` | polygon extraction, shape vectors (`core.py`, `fast_features.py`), reference and aerial tasks |
| `pipelines/gen_test_data.py` | cuts the test images |
| `tools/`, `run_nofly_pipeline.py` | draw or edit the no-fly zone (`data/references/non_fly_zone.csv` is included) |
| `get_Global_R_max.py`, `get_px_per_meter.py` | radius and image scale tools |
| `run_examples.py`, `run_examples.bat` | Examples 4.2 and 4.3 |
| `run_matching_all_v3.py` | candidate selection and evaluation |
| `plot_reference_maps.py` | Figures 3 and 6 |
| `plot_descriptor.py` | Figure 4 |
| `plot_localization.py` | Figure 5 and one map per test image |

Image scale: 2.445496 px/m.

## Contact and license

Prof. Phan Thanh An (thanhan@hcmut.edu.vn) and Tran Anh Dung
(trananhdung@iuh.edu.vn or tadung.sdh231@hcmut.edu.vn), Institute of Mathematical and Computational
Sciences (IMACS). The code is released under the MIT License (`LICENSE`).
