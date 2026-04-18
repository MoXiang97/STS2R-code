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
  Main scripts for synthetic data generation, preprocessing, training, ablation studies, and benchmark evaluation.

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

After downloading the dataset from Zenodo, please place the required files under the expected local directories described below, or update the paths in the scripts accordingly.

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

## Important note for users who download the GitHub ZIP archive

If you use **Download ZIP** from GitHub instead of cloning the repository, the extracted contents may contain an additional outer directory level.

Please make sure you run all scripts from the **actual project root**, i.e., the directory that directly contains:

- `assets/`
- `scripts/`
- `src/`
- `modules/`
- `utils/`

Some public scripts use **relative paths** such as `assets/Base_Perfect_scan_60CAD`.  
If you run Python from an outer folder instead of the actual project root, you may encounter errors such as:

```text
Error: Input directory assets\Base_Perfect_scan_60CAD not found.
```

In such cases, please first change into the correct project root directory and then run the scripts again.

This is a **working-directory issue**, rather than a code error or a problem with the dataset itself.

**Quick check:** before running any script, confirm that your current directory directly contains `assets/` and `scripts/`.

## Usage

This repository supports both:

- **one-click pipeline execution**
- **step-by-step generation and debugging**

### 1. One-click synthetic-data generation pipeline

To run the full virtual-data generation workflow in one command:

```bash
python scripts/00_Run_Data_Generation_Pipeline.py
```

This script is the **recommended default workflow** for most users. It generates the four synthetic-data variants used in the ablation design in a unified pipeline. The final variant corresponds to the released **04_STS2R** synthetic branch.

Please note that the full synthetic-data generation process may take a relatively long time (potentially hours, depending on hardware and environment), but it is the most convenient way to reproduce the complete generation workflow consistently.

### 2. Optional fast path for benchmark reproduction

For users who want to reproduce the released benchmark setting more quickly, the released synthetic data from Zenodo may be used directly instead of regenerating the final synthetic branch from scratch.

In this case, place the Zenodo synthetic data under:

```text
outputs/Generated_ablation_data/04_STS2R/
```

Then run the following scripts separately:

```bash
python scripts/05_ROI_Filter_and_NPY_Converter.py
python scripts/06_Generate_Stage1.5_Offline_Data.py
python scripts/run_benchmark.py
```

When running `run_benchmark.py`, set the experiment mode to:

```python
MODE = 'full'
```

Available modes are:

- `'full'`      - full-data training (Sim + Real)
- `'real_only'` - real-data-only training
- `'sim_only'`  - synthetic-data-only training
- `'all'`       - runs all of the above sequentially

Using the Zenodo synthetic data directly is acceptable if the goal is to reproduce the **final released benchmark setting** more efficiently.  
However, if the goal is to reproduce the **entire synthetic-data generation process** and all ablation variants in a consistent way, we recommend running the full `00_Run_Data_Generation_Pipeline.py` pipeline.

### 3. Step-by-step synthetic-data generation

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

### 4. Preprocessing and offline-data generation

The repository also provides the following intermediate scripts for preprocessing and benchmark preparation:

#### ROI filtering and NPY conversion

```bash
python scripts/05_ROI_Filter_and_NPY_Converter.py
```

This script performs ROI filtering and converts the generated data into the `.npy` format required by later steps.

#### Stage 1.5 offline-data generation

```bash
python scripts/06_Generate_Stage1.5_Offline_Data.py
```

This script generates the Stage 1.5 offline data used by the subsequent benchmark pipeline.

### 5. Run ablation experiments

To reproduce the ablation experiments reported in the paper:

```bash
python scripts/run_ablation.py
```

This script is used for evaluating progressive synthetic data-generation variants and their impact on downstream real-world transfer performance.

### 6. Run benchmark experiments

To reproduce the benchmark experiments reported in the paper:

```bash
python scripts/run_benchmark.py
```

This script is used for the complementary Stage 2 training settings described in the paper, including:

- Real-only
- Synthetic-only
- Synthetic + limited real

Before running this script, please check the `MODE` setting in the file according to your intended benchmark configuration.

## Reproducibility Notes

- The released dataset is stored separately on Zenodo.
- Update dataset paths before running the scripts if your local directory structure differs from the default assumptions.
- The one-click pipeline (`00_Run_Data_Generation_Pipeline.py`) is the recommended default workflow for complete reproduction.
- For faster reproduction of the final released setting, the Zenodo synthetic data may be placed directly under `outputs/Generated_ablation_data/04_STS2R/`, followed by `05_ROI_Filter_and_NPY_Converter.py`, `06_Generate_Stage1.5_Offline_Data.py`, and `run_benchmark.py`.
- Training outputs, logs, checkpoints, and generated experiment results are not included in this public repository.
- Running the generation and experiment scripts will automatically create the required output files and folders locally.
- Some scripts are intended for paper reproduction and may assume specific directory conventions or checkpoint locations.

## Recommended Workflow

For most users, the recommended order is:

1. Download the dataset from Zenodo.
2. Place the required Zenodo files under the expected local directories described above, or update the paths in the scripts if needed.
3. Confirm that you are running from the actual project root directory.
4. Run:
   - `00_Run_Data_Generation_Pipeline.py` for one-click generation, or
   - `01_Generate_V_Base.py` to `04_Generate_STS2R_Sim.py` step by step for debugging and inspection.
5. If using the Zenodo synthetic shortcut for the final released setting, place the released synthetic data under:
   - `outputs/Generated_ablation_data/04_STS2R/`
6. Run:
   - `05_ROI_Filter_and_NPY_Converter.py`
   - `06_Generate_Stage1.5_Offline_Data.py`
7. Run:
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
