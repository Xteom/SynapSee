# SynapSee

SynapSee is an innovative project exploring the use of EEG signals for image categorization. This repository contains all the necessary code and data for replicating and extending our research.

## How to clone this repository

```bash
git clone --recursive-submodules https://github.com/NeuroLab-ITAM/SynapSee.git
```

or

```bash
git clone http://github.com/NeuroLab-ITAM/SynapSee.git
cd SynapSee_data
git submodule init
git submodule update
```

## Code structure

* databuilder.ipynb: Notebook to build the dataset from the raw data.
* SynapSee_exp.ipynb: Notebook to run the experiments.
* SynapSee_exp.py: Python file to run the experiments.
* SynapSee_eda.ipynb: Notebook to explore the dataset csv.
* SynapSee_eda-II.ipynb: Notebook to explore the dataset from json.
* /images: Folder with the images used in the experiments.
* /SynapSee_data: Folder with the dataset.
* LSTM-CNN.ipynb: Notebook with the LSTM-CNN model
* SynapSee_models.ipynb: Notebook where we are currently working on new models.
