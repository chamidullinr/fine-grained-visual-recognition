# Fine-grained Visual Recognition with Side Information

## Overview
This repository contains supplementary material to my Master's thesis -
**Fine-grained Visual Recognition with Side Information**.

The thesis presents a method for fine-grained visual snake and fungi species recognition with side information.
The proposed method is based on state-of-the-art deep neural networks for classification: Convolutional Neural Networks and Vision Transformers.
The performance improvements are achieved by:
1) adopting loss functions proposed to address the class imbalance;
2) adjusting predictions by prior probabilities of side information like location and time of observation;
3) applying a weakly supervised method to localize snakes and fungi in images and crop the images based on the detected regions to enrich the training data.


## Content
### Cleaned SnakeCLEF Metadata
* [Cleaned training set](data/snake_clef2021_dataset/SnakeCLEF2021_train_metadata_cleaned.csv)
* [Cleaned reduced training set](data/snake_clef2021_dataset/SnakeCLEF2021_train_metadata_mini.csv)
* [Cleaned test set](data/snake_clef2021_dataset/SnakeCLEF2021_test_metadata_cleaned.csv)

### SnakeCLEF Additional Data
* [Country-species presence mapping](data/snake_clef2021_dataset/species_to_country_mapping.csv)
* [Medical importance of species](data/snake_clef2021_dataset/species_medical_importance.csv)

### Detected Bounding Boxes using Saliency-based localization method
* [SnakeCLEF dataset](03_informed_augmentation/data_snake/SnakeCLEF_bbox_annotations.csv)
* Danish Fungi dataset - [DF20](03_informed_augmentation/data_fungi/DF_bbox_annotations.csv) and [DF20M](03_informed_augmentation/data_fungi/DFM_bbox_annotations.csv)



### Python Scripts and Jupyter Notebooks
* Training and testing on the snake species recognition task:
  * [Training script](train_snake.py)
  * [Testing script](test_snake.py)
  * [Training Notebook](train_snake.ipynb)
  * [Testing Notebook](test_snake.ipynb)
  * [Training script](train_snake_crop.py) on cropped images created using saliency-based localization method
* Training and testing on the fungi species recognition task:
  * [Training script](train_fungi.py)
  * [Testing script](test_fungi.py)
  * [Training Notebook](train_fungi.ipynb)
  * [Testing Notebook](test_fungi.ipynb)
  * [Training script](train_fungi_crop.py) on cropped images created using saliency-based localization method
* [Data Preparation](01_data_preparation) - notebooks for preparation, exploration, and cleaning of the SnakeCLEF and Danish Fungi datasets.
* [Side Information](02_side_information) - notebooks for metadata inclusion.
  On the SnakeCLEF dataset, the method drops the predictions of the species not occurring in the country of the given image.
  For fungi species recognition, the method calibrates and adjusts the predictions by the prior probabilities of side information like habitat, substrate, location, and time of observation.
* [Informed Augmentation](03_informed_augmentation) - notebooks for applying a weakly supervised saliency-based method to localize snakes and fungi in images.
* [Venomous/Non-venomous Snake Classification ](04_venomous_classification) - example of using the proposed method to decide on medical response to snake bites.
* [Training Results](experiment_results)

## Getting Started
### Datasets
The snake and fungi datasets, used in this thesis, are publicly available at:
* [SnakeCLEF 2021](https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge)
* [Danish Fungi 2020](https://sites.google.com/view/danish-fungi-dataset)

### Package Dependencies
The proposed method wes developed using `Python=3.8` with `PyTorch=1.7.1` machine learning framework.
The pre-trained CNN networks were used from PyTorch Image Models library `timm=0.4.12`,
and the pre-trained Vision Transformers were used from Hugging Face Trasformers library `transformers=4.12.3`.
Additionally, the repository requires packages:
`numpy`, `pandas`, `scikit-learn`, `matplotlib` and `seaborn`.

To install required packages with PyTorch for CPU run:
```bash
pip install -r requirements.txt
```

For PyTorch with GPU run:
```bash
pip install -r requirements_gpu.txt
```

The requirement files do not contain `jupyterlab` nor any other IDE.
To install `jupyterlab` run
```bash
pip install jupyterlab
```

## Authors
**Rail Chamidullin** - chamidullinr@gmail.com  - [Github account](https://github.com/chamidullinr)
