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

### Custom CNN Models

| Network Architecture                                                                                                                                                       | Accuracy | Precision | Recall | F1 Score |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|-----------|--------|----------|
| 2x2x64 conv, 2x2 max pool, 3x3x128 conv, 2x2 max pool, 4x4x256 conv, 2x2 max pool, 4x4x512 conv, 2x2 max pool, 2x2x512 conv, 25088x1024 FC, 1024x512 FC, 512x1 FC   | 0.6918   | 0.7215    | 0.7125 | 0.7170   |
| 3x3x16 conv, 2x2 max pool, 3x3x16 conv, 2x2 max pool, 4x4x32 conv, 2x2 max pool, 3x3x64 conv, 2x2 max pool, 2x2x256 conv, 16384x512 FC, 512x64 FC, 64x1 FC         | 0.7260   | 0.7703    | 0.7125 | 0.7403   |
| 3x3x64 conv, 2x2 max pool, 3x3x128 conv, 2x2 max pool, 3x3x256 conv, 2x2 max pool, 3x3x512 conv, 2x2 max pool, 3x3x512 conv, 12800x1024 FC, 1024x512 FC, 512x1 FC   | 0.6781   | 0.7260    | 0.6625 | 0.6928   |
| 2x2x128 conv, 2x2 max pool, 2x2x256 conv, 2x2 max pool, 2x2x256 conv, 2x2 max pool, 2x2x512 conv, 2x2 max pool, 2x2x1024 conv, 2x2 max pool, 2x2x1024 conv, 16384x1024 FC, 1024x512 FC, 512x64 FC, 64x1 FC | 0.5685   | 0.5726    | 0.8375 | 0.6802   |

### Transfer Learning (VGG-19)

| Network Architecture (Classifier)                             | Accuracy | Precision | Recall | F1 Score |
|---------------------------------------------------------------|----------|-----------|--------|----------|
| 25088x1024 FC, 1024x512 FC, 512x1 FC                         | 0.830    | 0.84      | 0.85   | 0.84     |
| 25088x1024 FC, 1024x64 FC, 64x1 FC                           | 0.607    | 0.63      | 0.67   | 0.65     |
| 25088x2048 FC, 2048x128 FC, 128x1 FC                         | 0.5479   | 0.5479    | 0.99   | 0.708    |

*Note: FC refers to fully connected layers.*

## Acknowledgements
* [UCSD-AI4H/COVID-CT](https://www.kaggle.com/datasets/luisblanche/covidct?resource=download) for the dataset.
* The authors of the arXiv [paper](https://arxiv.org/abs/2003.13865) for their research and dataset compilation.
