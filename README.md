# Title

MilCamo: A Lightweight Transformer-Based Framework for Human Camouflage Detection


# Abstract

A great degree of similarity between target object and background in an image remains on in every of the most difficult tasks in computer vision. This paper proposed a new technique of segmentation of human-imported objects, backbone of this methodology is Global Context Visual Transformer (GCVT). To address the complex aspects of camouflage like occlusion, mixing with background, color similarity; proposed model incorporates advanced concepts such as patch embedding (vertical) mix with powerful encoders, global query mechanisms and characterize smoothing feature’s module. Decoder has been incorporated with cross and self attributes, pathing and pass-through to restore articulated edges of objects gradually. ..........
# Requirements
tensorflow==2.17.0
keras==3.5.0
numpy
opencv-python
scipy
matplotlib
pandas
tqdm
scikit-image
scikit-learn
albumentations
pip install -r requirements.txt
# Installation
The TensorFlow framework was used to create and construct our model, which was then run on the TPUv5 provided by Google Colab.
consult Reproducibility / REPRODUCE.md 
# Dataset Preparation
Participating photos were stored on Google Cloud bucket and standardized to a resolution of 224 × 224 pixels for training, which is regarded as a machine learning sweet spot.

# Datasets 
for DATASET DETAILS   DATASETS/DATASET.md

https://www.kaggle.com/datasets/ivanomelchenkoim11/camo-dataset

https://www.kaggle.com/datasets/ismailelomarialaoui/cod10k

www.polsl.pl/rau6/chameleon-database-animal-camouflage-analysis/https://www.kaggle.com/datasets/ivanomelchenkoim11/nc4k-dataset

https://www.kaggle.com/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k

Military - Hidden Human - Dataset - Google Drive
(https://drive.google.com/drive/u/2/folders/1V8TdUEiUQDa3SZGTATRFXTmlvq0dms5A)

a new dataset named HuCOD (Human COD) is created, which is comprised of 2 general categories: M_ denotes Military, and NM_ denotes Non-Military humans, with the same image name as in the original dataset. E.g., M_ACD1K_image517.jpg shows the following pattern: M_ indicates military-specific, and the dataset name and image name are the same as in the original dataset.
#  Procedure
see models/diagram.md
# Results
see results/ readme.md

Dataset Splits:
The exact training, validation, and testing partitions used in the experiments are provided in the Datasets directory through the files:
- train.txt
- val.txt
- test.txt

These files contain the complete image lists used to reproduce the experimental results reported in the manuscript.

# Citation
----
# Contact Information
Khurram shehzad : Khurram.ch06@gmail.com / Arizon45@yahoo.com

Saeed ur Rehman : srehman@ciitwah.edu.pk

Muhammad Fayyaz : fayyaz.uos@gmail.com

