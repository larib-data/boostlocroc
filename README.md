# boostlocroc: automated detection of loss and return of consciousness based on electro-encephalograms

This repo contains a Python package. It implements an off-the-shelf method to automatically detect times of loss of consciousness (LoC) and return of consciousness (RoC) on intraoperative EEG during general anesthesia. 

The detection is made by fitting an activation function to the outputs of the gradient boosting model trained on the PSD of 30-second epochs of EEG signal, see [paper](https://ieeexplore.ieee.org/abstract/document/10199018). 


## Installation
First, ensure you have downloaded the boost-loc-roc folder containing this README and have an up-to-date version of ```pip```. To install the ```boostlocroc``` Python package, run the following command lines in a terminal:
```
cd <this_folder>
pip install -e .
```
and follow the instructions. Once that's done, you can import and use the package functions in your Python files; see ```quickstart.ipynb``` for an example.

## Quickstart
See `quickstart.ipynb` on how to load the trained model on an EEG recording taken from the [VitalDB](https://vitaldb.net/) dataset. The expected data type is .fif.

## Project structure
The main functions are in the `boostlocroc/main.py` file. The code for extracting EEG features before applying the model is in the `boostlocroc/eeg_features.py` file. The code for visualization is in the `boostlocroc/vis.py` file. \
The `boostlocroc/model/` folder contains the models used for prediction. The ONNX model (`voting_model.onnx`) is primary and used off-the-shelf. The scikit-learn models are not used off-the-shelf and are kept for archival purposes.
They could be helpful if you try to train your own model. \
The `boostlocroc/archive/` folder contains code used for the [paper](https://ieeexplore.ieee.org/abstract/document/10199018) and is now deprecated. This code is just for information and is not expected to run smoothly if you run it. \

## Citing this work
If this package was helpful to you, please cite it: \
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
