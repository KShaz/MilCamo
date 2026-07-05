# Environment

pip install -r requirements.txt

# Training

python train.py

# Testing

python test.py \
    --weights weights/MilCamo.h5

# Evaluation

python evaluate.py \
    --gt_dir dataset/Test/GT \
    --pred_dir predictions/

# Dataset splits

train.txt
val.txt
test.txt

# Hardware

Google TPU v3-8
TensorFlow version
CUDA version
