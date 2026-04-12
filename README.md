# STS2R (Sim-to-Real Shoe Point Cloud Segmentation)

This repository contains the code for the **STS2R** project, including synthetic data generation, preprocessing, model training, ablation studies, and benchmark evaluation for robotic shoe-upper processing boundary segmentation.

The code supports the accompanying STS2R dataset paper and its Sim2Real experiments on multimodal 3D point clouds.

## Project Structure

- `assets/`  
  Reusable textures and pattern assets used for synthetic-data generation and appearance augmentation.

- `checkpoints/`  
  Optional pre-trained model weights.

- `modules/`  
  Core modules for data augmentation, I/O, and supporting functions.

- `scripts/`  
  Main scripts for synthetic data generation, training, ablation studies, and benchmark evaluation.

- `src/`  
  Model architectures, loss functions, and training-related source code.

- `utils/`  
  Common utilities and helper functions.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MoXiang97/STS2R-code.git
cd STS2R-code
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment note

This project requires Python and the packages listed in `requirements.txt`.

Please ensure that your local PyTorch installation matches your CUDA environment if GPU training is intended.

## Dataset

The **STS2R dataset** is released separately on Zenodo:

**Zenodo DOI:** https://doi.org/10.5281/zenodo.19528228

> This GitHub repository does **not** contain the complete released dataset.  
> For the full dataset release, please download the corresponding files from the Zenodo record above.

After downloading the dataset from Zenodo, place it under `./data/` or update the dataset paths in the scripts accordingly.

A typical local structure is:

```text
data/
├── canonical_virtual_source_templates/
├── synthetic_data/
├── real_test_data/
├── generation_assets/
└── STS2R_metadata.csv
```

## Directory mapping for Zenodo data

The complete STS2R dataset is distributed through Zenodo, while this GitHub repository provides the accompanying codebase only.

Some scripts in this repository expect specific local directory names. After downloading and extracting the Zenodo release, please place the corresponding files under the following paths in the GitHub project directory.

### Required local paths

- Zenodo `canonical_virtual_source_templates/`  
  → place under:
  `assets/Base_Perfect_scan_60CAD/`

- Zenodo real-data files required by the public code  
  → place under:
  `assets/Real_Data/Real_ShoeA/`  
  `assets/Real_Data/Real_ShoeB/`  
  `assets/Real_Data/Real_ShoeC/`

- Zenodo `generation_assets/`  
  → place under the corresponding local assets directory used by the scripts.

If your local folder names differ, please update the paths in the scripts accordingly.

The GitHub repository intentionally does not include the complete data-dependent folders.  
For full reproduction, users should download the corresponding files from the Zenodo release and restore them under the expected local directory structure.

## Usage

This repository supports both:

- **one-click pipeline execution**
- **step-by-step generation and debugging**

### 1. One-click synthetic-data generation pipeline

To run the full virtual-data generation workflow in one command:

```bash
python scripts/00_Run_Data_Generation_Pipeline.py
```

This script is intended for users who want to generate the required synthetic data in the default full pipeline.

### 2. Step-by-step synthetic-data generation

The following scripts correspond to progressively enriched synthetic data variants used in the paper. They are useful for debugging, intermediate inspection, and ablation-oriented regeneration.

#### Base generation

```bash
python scripts/01_Generate_V_Base.py
```

Generates the **Base** synthetic data before geometric diversification, sensor-aware degradation, and full appearance generalization.

#### Geometric diversification

```bash
python scripts/02_Generate_V_Geo.py
```

Generates the **Base + Geo** variant by introducing geometric diversification.

#### Sensor-aware degradation

```bash
python scripts/03_Generate_V_Phys.py
```

Generates the **Base + Geo + Phys** variant by adding sensor-aware degradation.

#### Final STS2R synthetic branch

```bash
python scripts/04_Generate_STS2R_Sim.py
```

Generates the final released synthetic branch, corresponding to **Base + Geo + Phys + App**.

### 3. Run ablation experiments

To reproduce the ablation experiments reported in the paper:

```bash
python scripts/run_ablation.py
```

This script is used for evaluating progressive synthetic data-generation variants and their impact on downstream real-world transfer performance.

### 4. Run benchmark experiments

To reproduce the benchmark experiments reported in the paper:

```bash
python scripts/run_benchmark.py
```

This script is used for the complementary Stage 2 training settings described in the paper, including:

- Real-only
- Synthetic-only
- Synthetic + limited real

## Reproducibility Notes

- The released dataset is stored separately on Zenodo.
- Update dataset paths before running the scripts if your local directory structure differs from the default assumptions.
- Training outputs, logs, checkpoints, and generated experiment results are not included in this public repository.
- Running the generation and experiment scripts will automatically create the required output files and folders locally.
- Some scripts are intended for paper reproduction and may assume specific directory conventions or checkpoint locations.

## Recommended Workflow

For most users, the recommended order is:

1. Download the dataset from Zenodo.
2. Configure dataset paths if needed.
3. Run:
   - `00_Run_Data_Generation_Pipeline.py` for one-click generation, or
   - `01_Generate_V_Base.py` to `04_Generate_STS2R_Sim.py` step by step for debugging and inspection.
4. Run:
   - `run_ablation.py` for ablation experiments
   - `run_benchmark.py` for benchmark experiments

## Notes on Data and Labels

The released dataset uses a unified 8-column ASCII `.txt` point-cloud format:

1. `X`
2. `Y`
3. `Z`
4. `R`
5. `G`
6. `B`
7. `Label1`
8. `Label2`

For downstream segmentation:

- `Label1` is the primary segmentation target.
- In the real-world branch, `Label2` should be ignored.
- In the synthetic branch, `Label2` is a panel-structural identifier and should not be interpreted as boundary-level ground truth after boundary reconfiguration.

## License

This **code repository** is released under the **MIT License**.

The **STS2R dataset** is distributed separately on Zenodo under **CC BY 4.0**.

Please check the Zenodo record for dataset-specific licensing terms and usage conditions.

## Citation

If you use the **STS2R dataset**, please cite the Zenodo record:

**STS2R dataset**  
Zenodo DOI: https://doi.org/10.5281/zenodo.19528228

If you use this **code repository**, please also cite the accompanying paper after publication.

## Contact

For issues related to code usage, reproducibility, or repository setup, please open an issue in this GitHub repository.
