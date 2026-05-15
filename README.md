# Cross-Domain Contrastive Learning for Time Series Clustering

This repository contains the source code for reproducing experiments of the paper entitled:

**"Cross-Domain Contrastive Learning for Time Series Clustering"**

---

## Table of Contents

- [Datasets](#datasets)
- [Installation](#installation)
- [Setting Parameters](#setting-parameters)
- [Run](#run)
- [Contact](#contact)
- [File Structure](#file-structure)
- [Citation](#citation)

---

## Datasets

The experiments use datasets from the **UCR Time Series Classification Archive**:  
http://www.timeseriesclassification.com/

To save space and make reading easier, we have converted the datasets into CSV format and compressed them. The processed data is available at:  
https://pan.baidu.com/s/1XeKQUaSPyENEp3SDS3q0sA (password: `AK47`)

---

## Installation

You can install the required packages using your favorite package manager or create a new Python environment (Python 3.6 or higher).  

```
pip install -r requirements.txt
```

---

## Setting Parameters

All hyperparameters are set in `config/CDCC.yaml`.  

Hyperparameters were optimized using grid search and the Adam optimizer. The main parameters include:

- **Learning rate** (`lr`): {0.01, 0.001, 0.0003}  
- **Number of BiLSTM layers** (`num_layers`): {1, 2, 3}  
- **Batch size** (`batch_size`): dependent on dataset size, searched from {8,16,32,64,128,256}  
- **Dropout rate** (`p`): {0.1, 0.3, 0.5}  

Make sure to adjust these parameters according to your experiments.

---

## Run

To train the model and reproduce the experiments:

```
python main.py -f config/CDCC.yaml
```

---

## Contact

If you have any questions or encounter any issues, please feel free to contact me via email:  

**luojike1418743@gmail.com**  

I will be happy to help!

---

## File Structure

A brief overview of the main folders and files:

```
algorithm/             # Core algorithms and model code
config/                # Configuration files (e.g., CDCC.yaml)
data/                  # Dataset directory (UCR CSV files)
main.py                # Entry point for training and testing
requirements.txt       # Python dependencies
utils/                 # Utility scripts (data loading, metrics, etc.)
README.md              # This file
```

---

## Citation

If you use this code for your research, please cite our paper:

```
@article{Peng_Luo_Lu_Wang_Li_2024,
  title={Cross-Domain Contrastive Learning for Time Series Clustering},
  volume={38},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/28740},
  DOI={10.1609/aaai.v38i8.28740},
  abstractNote={Most deep learning-based time series clustering models concentrate on data representation in a separate process from clustering. This leads to that clustering loss cannot guide feature extraction. Moreover, most methods solely analyze data from the temporal domain, disregarding the potential within the frequency domain. To address these challenges, we introduce a novel end-to-end Cross-Domain Contrastive learning model for time series Clustering (CDCC). Firstly, it integrates the clustering process and feature extraction using contrastive constraints at both cluster-level and instance-level. Secondly, the data is encoded simultaneously in both temporal and frequency domains, leveraging contrastive learning to enhance within-domain representation. Thirdly, cross-domain constraints are proposed to align the latent representations and category distribution across domains. With the above strategies, CDCC not only achieves end-to-end output but also effectively integrates frequency domains. Extensive experiments and visualization analysis are conducted on 40 time series datasets from UCR, demonstrating the superior performance of the proposed model.},
  number={8},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Peng, Furong and Luo, Jiachen and Lu, Xuan and Wang, Sheng and Li, Feijiang},
  year={2024},
  month={Mar.},
  pages={8921–8929}
}
