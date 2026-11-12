# Ovule3D source snapshot

This archive contains selected Python source for topology prediction, complete-volume instance reconstruction and cell morphometry in Arabidopsis ovule imaging. It includes the core package, single-case entry points, and selected low-label, native PlantSeg and different-organ analysis routines. It preserves the archived source files byte for byte.

## Inputs and environment

- PlantSeg Ovules source data: https://osf.io/w38uf/
- Cambridge apical stem-cell-niche source data: https://doi.org/10.17863/cam.7793
- Official PlantSeg Ovules checkpoint record: https://doi.org/10.5281/zenodo.7805434
- The implementation uses Python, PyTorch and compatible torchvision. The project requirement file lists several selected package versions. HDF5, NumPy, SciPy and scikit-image are needed for image and mesh processing. Optional native PlantSeg routines require PlantSeg and its graph-segmentation backend. Install dependencies separately under their respective terms.

## Entry points

From the extracted archive, inspect the command-line options using:

```sh
python data/download_plantseg_ovules.py --help
python scripts/prepare_data.py --help
python scripts/cache_features.py --help
python scripts/train.py --help
python scripts/evaluate_volume.py --help
python scripts/export_cell_models.py --help
```

Pass explicit input and output locations. The feature-cache script retains the historical Windows checkpoint default and accepts an explicit checkpoint path. Complete-volume evaluation uses a run directory containing the applicable model checkpoint and configuration. Use the paper Methods for the declared split, model, spacing convention and reconstruction settings.

The snapshot contains source code and environment notes. Raw images, annotations, checkpoints, feature caches, predictions, study records and the manuscript are acquired separately. This archive does not include the original orchestration environment, original historical source revisions or all inputs needed to rerun every reported experiment. No clean-machine or end-to-end replay was performed for this release candidate.

SHA256SUMS lists the checksums of the included project files.
