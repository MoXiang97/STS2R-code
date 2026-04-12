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