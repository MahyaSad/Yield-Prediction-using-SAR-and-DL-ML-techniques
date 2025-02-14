# Yield Prediction using SAR with Patch-Based 3D-CNN and ML Methods

Overview

This repository contains the Python code, trained models, and datasets used in our research on crop yield prediction for winter wheat, soybeans, and corn in Michigan (MI) using Synthetic Aperture Radar (SAR) observations. Our research has been published in Computers and Electronics in Agriculture. The study explores the potential of SAR data combined with deep learning (DL) and machine learning (ML) models for accurate crop yield estimation.

Project Description

The goal of this project is to estimate crop yields using SAR data and advanced machine learning techniques. We employed patch-based 3D-Convolutional Neural Networks (3D-CNNs) and other ML methods, such as Random Forest (RF), Support Vector Machine (SVM), and eXtreme Gradient Boosting (XGBoost), to predict crop yields from Sentinel-1 SAR observations.

Key contributions of this study include:

Utilizing multi-temporal Sentinel-1 SAR data for yield prediction.

Exploring the impact of different feature combinations on ML and DL models.

Assessing the influence of SAR resolution (30m vs. 50m) on prediction accuracy.

Demonstrating the effectiveness of 3D-CNNs in mitigating noise and improving yield estimation.

Repository Contents

1. Code

ML.py: Contains implementations of Random Forest, SVM, and XGBoost models.

patch-based_3D-CNNs.py: Implementation of the 3D-CNNs deep learning model for yield prediction.

train.py: Training scripts for ML and DL models.

evaluate.py: Evaluation scripts for model performance assessment.

utils.py: Utility functions for data preprocessing and feature extraction.

2. Trained Models

best_model_patch3D-CNN_test.pth: Pre-trained 3D-CNNs model.

Saved models of ML-based yield predictions using different feature combinations.

3. Datasets

all_samples_30.rar: SAR and reference data at 30-meter resolution.

all_samples_50.rar: SAR and reference data at 50-meter resolution.

Processed Sentinel-1 VH and VV data.

Climate data including precipitation, minimum, and maximum temperature.

Crop yield reference data from John Deere yield mapping.



