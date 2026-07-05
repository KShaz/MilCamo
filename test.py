# test.py

import os
import tensorflow as tf
from cos_eval_tf_metrics import (
    SScore,
    ESimilarityMetric,
    WeightedFScoreMetric
)

from losses.losses import NewMyLoss
from models.milcamo import segmentationModelFunction


def preprocess_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (224, 224))
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask


def create_test_dataset(image_paths, mask_paths, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, mask_paths)
    )
    dataset = dataset.map(preprocess_image)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_model(weights_path):
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={
            "NewMyLoss": NewMyLoss,
            "SScore": SScore,
            "ESimilarityMetric": ESimilarityMetric,
            "WeightedFScoreMetric": WeightedFScoreMetric,
        },
    )
    return model


def run_test(model, dataset):
    results = model.evaluate(dataset)
    print(results)
    return results


if __name__ == "__main__":
    print("MilCamo Test Pipeline")