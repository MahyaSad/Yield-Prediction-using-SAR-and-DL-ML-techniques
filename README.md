Yield Prediction using SAR with Patch-Based 3D-CNN and ML Methods

This repository contains the Python code, trained models, and datasets used in our research on yield prediction for winter wheat, soybeans, and corn in Michigan (MI) using Synthetic Aperture Radar (SAR) observations. The research has been published in the journal Electronics and Computers in Agriculture. The paper can be accessed via DOI.
Project Overview

The goal of this project is to estimate crop yields using SAR data and machine learning techniques. We employed patch-based 3D-Convolutional Neural Networks (3D-CNNs) combined with other machine learning (ML) methods to predict yields for winter wheat, soybeans, and corn.
Contents

This repository includes the following:

    Python Code: Scripts for training and evaluating ML models, including patch-based 3D-CNNs.
    Trained Models: Saved models of patch-based 3D-CNNs trained on the combined dataset of all crops.
    Datasets: SAR and reference data for winter wheat, soybeans, and corn covering the years 2016-2023.

Getting Started
Prerequisites

To run the code in this repository, you will need the following:

    Python 3.x
    Required Python packages listed in requirements.txt

Installation

    Clone this repository:

    bash

git clone https://github.com/yourusername/yield-prediction-sar.git
cd yield-prediction-sar

Install the required packages:

bash

    pip install -r requirements.txt

Usage
Training the Model

To train the patch-based 3D-CNN model, run:

bash

python train_3dcnn.py --data_dir path/to/data --model_dir path/to/save/model

Evaluating the Model

To evaluate the trained model, run:

bash

python evaluate_3dcnn.py --data_dir path/to/data --model_dir path/to/model

Predicting Yields

To predict yields using the trained model, run:

bash

python predict_yields.py --data_dir path/to/data --model_dir path/to/model

Data

The dataset used in this project includes SAR observations and reference data for winter wheat, soybeans, and corn from 2016 to 2023. The data is organized by crop and year.
Repository Structure

arduino

yield-prediction-sar/
│
├── data/
│   ├── wheat/
│   ├── soybeans/
│   ├── corn/
│
├── models/
│   ├── 3dcnn_model.h5
│
├── scripts/
│   ├── train_3dcnn.py
│   ├── evaluate_3dcnn.py
│   ├── predict_yields.py
│
├── requirements.txt
├── README.md

Citation

If you use this code or data in your research, please cite our paper:

Mahya Ghazi Zadeh Hashemi, et al. (2024). "Yield Prediction Using SAR with Patch-Based 3D-CNN and ML Methods." Electronics and Computers in Agriculture. DOI: link.
Acknowledgments

This research was supported by [Funding Agency]. We thank [Collaborators/Institutions] for their contributions.
License

This project is licensed under the MIT License - see the LICENSE file for details.
