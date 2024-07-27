# COVID-19 Lung CT Scan Classification

## Overview
This project aims to develop a deep learning classifier to predict the presence of COVID-19 from lung CT scans. The dataset used consists of CT images with and without COVID-19, curated from various medical publications. The project explores both custom convolutional neural networks (CNNs) and transfer learning using pre-trained models like VGG-19 and ResNet-50. The evaluation is performed using 5-fold cross-validation, ensuring robust performance metrics.

## Dataset
The dataset contains CT images categorized into COVID-19 positive and negative cases. It is available on [Kaggle](https://www.kaggle.com/datasets/luisblanche/covidct?resource=download) and is collected from COVID-19-related papers. For more information about the dataset and its curation process, refer to the arXiv [paper](https://arxiv.org/abs/2003.13865s).

## Methodology
1. Custom CNN Model
Architecture: Developed using PyTorch with convolutional and pooling layers.
Optimization: Experimented with different architectures and hyperparameters to achieve the best accuracy.
Overfitting Prevention: Techniques like normalization, dropout, and early stopping were employed.
2. Transfer Learning
Pre-trained Models: Used models such as VGG-19 and ResNet-50.
Fine-tuning: Retrained the last few layers using the dataset.
Overfitting Prevention: Similar techniques as above were used.

## Evaluation Metrics
Precision, Recall, and F1 Score: Calculated for each model to assess performance.
5-fold Cross-Validation: Ensured robust evaluation.

## Acknowledgements
* [UCSD-AI4H/COVID-CT](https://www.kaggle.com/datasets/luisblanche/covidct?resource=download) for the dataset.
* The authors of the arXiv [paper](https://arxiv.org/abs/2003.13865) for their research and dataset compilation.
