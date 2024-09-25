# BrainIB+

## Interpretable Diagnosis of Schizophrenia Using Graph-Based Brain Network Information Bottleneck

This is the BrianIB+ demo for the BSNIP and UCLA dataset, BrianIB+ is also known as BrainIB V2.

This README.md is for all readers.
 
### Contents
- [Getting Started](#getting-started)
  - [Configuration requirements](#configuration-requirements)
  - [Step by step](#step-by-step)
- [File Directory Description](#file-directory-description)




### Getting Started

Please ensure that the device has a working environment。

#### Configuration requirements

1. Activate your own environment
2. Install packeages
```sh
pip install -r requirements.txt
```

#### **Step by step**

1. Clone the repo
```sh
git clone git@github.com:TianzhengHU/BrainIB_coding.git
```

2. Download the example dataset from Google Drive.
The datasets used in this project are public avaliable. To facilitate testing, we make the preprocessed dataset publicly available on Google drive.
Please download dataset files: **UCLA.mat** and **BSNIP.mat** from link:
```
 https://drive.google.com/drive/folders/1ca9-nxsldpN3Cam_bGX6odce1VlOt96J?usp=share_link).
```
Put them under the "real_data" folder.


3. Run the code
```sh
python SGSIB_main.py
```


### File Directory Description

```
BrainIB_GIB 
├── /SGSIB/
│  ├── /model/
│  ├── sub_graph_generator.py
│  ├── sub_node_generator.py
│  └── utils.py
├── /baseline_data/
│  ├── /data/
│  └── baseline_main.py
├── /real_data/
│  ├── /Utils_preprocessing/
│  │  ├── BrainNet_index_map.xlsx
│  │  ├── BSNIP_new_id.xlsx
│  │  ├── ICNs_v2.xlsx
│  │  ├── closest_map_points.xlsx
│  │  ├── UCLA_preprocessing.py
│  │  └── Map_groupICA_and_BrainNet_map.py
│  ├── analysis_data.py
│  └── create_dataset.txt
├── Graph_network_main.py
├── SGSIB_main.py
├── Traditional_main.py
├── Visualization.py
├── requirements.txt
└── README.md

```








how to run 
run Python file "SGSIB_main.py" to start our demo.
