# LoC-RoC: Automated detection of Loss and Return of Consciousness based on electro-encephalograms

This repo contains an off-the-shelf method to automatically detect times of loss of consciousness (LoC) and return of consciousness (RoC) on intraoperative EEG during general anesthesia. 

The detection is made by fitting an activation function to the outputs of a a gradient boosting model trained on the PSD of 30-seconds epochs of EEG signal, see [paper](https://ieeexplore.ieee.org/abstract/document/10199018). 


## Installation
First, ensure that you have downloaded the boost-loc-roc folder containing this README and that you have an up-to-date version of ```conda```. 

We recommend creating a dedicated virtual environment using ```conda```. This will install all necessary Python packages in their correct versions. 
```
cd <this_folder>
conda env create -n loc-roc --file environment.yml
```

## Quickstart
See `quickstart.ipynb` on how to load the trained model and apply it to your data. Expected data type is .fif. Useful functions for this task are in the `boost-loc-roc` folder. \
The `boost-loc-roc/utils/` folder contains code which is not needed for off-the-shelf use, but could be helpful in case you try to train your own model.

## Citing this work
If this package was useful to you, please cite it: \
O. S. Aubin, I. Khemir, J. Perdereau, C. Touchard, F. Vallée and J. Cartailler, "Repurposing electroencephalographic signal for automatic segmentation of intra-operative periods under general anesthesia," IEEE EUROCON 2023 - 20th International Conference on Smart Technologies, Torino, Italy, 2023, pp. 286-290, doi: 10.1109/EUROCON56442.2023.10199018.
```
@INPROCEEDINGS{aubin2023repurposing,
  author={Aubin, O. Saint and Khemir, I. and Perdereau, J. and Touchard, C. and Vallée, F. and Cartailler, J.},
  booktitle={IEEE EUROCON 2023 - 20th International Conference on Smart Technologies}, 
  title={Repurposing electroencephalographic signal for automatic segmentation of intra-operative periods under general anesthesia}, 
  year={2023},
  volume={},
  number={},
  pages={286-290},
  doi={10.1109/EUROCON56442.2023.10199018}}
  ```