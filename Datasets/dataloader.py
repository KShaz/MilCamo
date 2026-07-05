import tensorflow as tf

def parse(serialized, image_shape=(224, 224, 3), mask_shape=(224, 224, 1)):
    """
    Parse function to convert TFRecord back to image and mask tensors.
    Can include additional preprocessing if needed.
    """
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'image_filename': tf.io.FixedLenFeature([], tf.string),
        'mask_filename': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'image_channels': tf.io.FixedLenFeature([], tf.int64),
        'mask_channels': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    # Deserialize the tensors
    image = tf.io.parse_tensor(parsed_example['image'], out_type=tf.float32)
    mask = tf.io.parse_tensor(parsed_example['mask'], out_type=tf.float32)

    # Reshape to known dimensions
    image = tf.reshape(image, shape=image_shape)
    mask = tf.reshape(mask, shape=mask_shape)

    return image, mask

# Alternative parse function with additional preprocessing

def parse_with_augmentation(serialized, image_shape=(224, 224, 3), mask_shape=(224, 224, 1)):
    """
    Parse function with data augmentation capabilities for normalized images (0-1).
    """
    image, mask = parse(serialized, image_shape, mask_shape)

    # Random horizontal flip (apply to both image and mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Random vertical flip (apply to both image and mask)
    # if tf.random.uniform(()) > 0.5:
    #     image = tf.image.flip_up_down(image)
    #     mask = tf.image.flip_up_down(mask)

    # Random brightness (only for image, reduced delta for normalized images)
    # image = tf.image.random_brightness(image, max_delta=0.2)

    # # Random contrast (only for image, adjusted range for normalized images)
    # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # # Random saturation (only for image, adjusted range for normalized images)
    # image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    # # Random hue (only for image, reduced delta for normalized images)
    # image = tf.image.random_hue(image, max_delta=0.05)

    # Random rotation (apply to both image and mask)
    # Note: Using 90-degree rotations since arbitrary angle rotation is complex
    # if tf.random.uniform(()) > 0.7:
    #     k = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
    #     image = tf.image.rot90(image, k)
    #     mask = tf.image.rot90(mask, k)

    # Random crop and resize (apply to both image and mask)
    if tf.random.uniform(()) > 0.5:
        crop_factor = tf.random.uniform([], 0.8, 1.0, dtype=tf.float32)
        crop_size = tf.cast(tf.cast(image_shape[0], tf.float32) * crop_factor, tf.int32)

        # Get random crop coordinates
        offset_height = tf.random.uniform([], 0, image_shape[0] - crop_size + 1, dtype=tf.int32)
        offset_width = tf.random.uniform([], 0, image_shape[1] - crop_size + 1, dtype=tf.int32)

        # Apply same crop to both image and mask
        image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size, crop_size)
        mask = tf.image.crop_to_bounding_box(mask, offset_height, offset_width, crop_size, crop_size)

        # Resize back to original size
        image = tf.image.resize(image, image_shape[:2])
        mask = tf.image.resize(mask, mask_shape[:2])

    # Add noise (only to image, reduced stddev for normalized images)
    if tf.random.uniform(()) > 0.7:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
        image = image + noise

    # Ensure values remain in valid range [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    mask = tf.clip_by_value(mask, 0.0, 1.0)

    return image, mask

# Input Pipeline for Local TFRecords

def create_local_dataset(tfrecord_pattern, batch_size=16, shuffle_buffer=1000,
                        use_augmentation=False, repeat=False):
    """
    Create dataset from local TFRecord files.

    Args:
        tfrecord_pattern: Pattern to match TFRecord files (e.g., 'shard_*.tfrec')
        batch_size: Batch size for training
        shuffle_buffer: Buffer size for shuffling
        use_augmentation: Whether to apply data augmentation
        repeat: Whether to repeat the dataset indefinitely
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # Get all matching TFRecord files
    shards = tf.io.matching_files(tfrecord_pattern)
    shards = tf.random.shuffle(shards)
    shards = tf.data.Dataset.from_tensor_slices(shards)

    # Create dataset from shards
    dataset = shards.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=AUTOTUNE,
        deterministic=False
    )

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Parse the records
    parse_fn = parse_with_augmentation if use_augmentation else parse
    dataset = dataset.map(parse_fn, num_parallel_calls=AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

# Input Pipeline for Google Cloud Storage TFRecords

def create_gcs_dataset(gcs_pattern, batch_size=16, shuffle_buffer=1000,
                      use_augmentation=False, repeat=False):
    """
    Create dataset from TFRecord files stored in Google Cloud Storage.

    Args:
        gcs_pattern: GCS pattern to match TFRecord files
                    (e.g., 'gs://your-bucket/tfrec_files/*.tfrec')
        batch_size: Batch size for training
        shuffle_buffer: Buffer size for shuffling
        use_augmentation: Whether to apply data augmentation
        repeat: Whether to repeat the dataset indefinitely
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # Get all matching TFRecord files from GCS
    shards = tf.io.matching_files(gcs_pattern)
    shards = tf.random.shuffle(shards)
    shards = tf.data.Dataset.from_tensor_slices(shards)

    # Create dataset from shards with interleaving for better performance
    dataset = shards.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=AUTOTUNE,
        deterministic=False
    )

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Parse the records
    parse_fn = parse_with_augmentation if use_augmentation else parse
    dataset = dataset.map(parse_fn, num_parallel_calls=AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

# Training and Validation Dataset Creation

def create_train_val_datasets(train_pattern, val_pattern=None, batch_size=16,
                             train_shuffle_buffer=1000, val_shuffle_buffer=100,
                             use_augmentation=True):
    """
    Create training and validation datasets.

    Args:
        train_pattern: Pattern for training TFRecord files
        val_pattern: Pattern for validation TFRecord files (if None, uses train_pattern)
        batch_size: Batch size
        train_shuffle_buffer: Shuffle buffer size for training
        val_shuffle_buffer: Shuffle buffer size for validation
        use_augmentation: Whether to use augmentation for training data
    """

    # Training dataset with augmentation
    train_dataset = create_local_dataset(
        train_pattern,
        batch_size=batch_size,
        shuffle_buffer=train_shuffle_buffer,
        use_augmentation=use_augmentation,
    )

    # Validation dataset without augmentation
    if val_pattern:
        val_dataset = create_local_dataset(
            val_pattern,
            batch_size=batch_size,
            shuffle_buffer=val_shuffle_buffer,
            use_augmentation=False,
        )
    else:
        # Use same pattern but without augmentation for validation
        val_dataset = create_local_dataset(
            train_pattern,
            batch_size=batch_size,
            shuffle_buffer=val_shuffle_buffer,
            use_augmentation=False,
        )

    return train_dataset, val_dataset

# Example usage:

# For local TFRecord files:
# train_dataset = create_local_dataset('shard_*.tfrec', batch_size=8, use_augmentation=True)

# For GCS TFRecord files:
train_dataset = create_gcs_dataset('gs://tensorflow-colab-tpu/train_tfrec_files/*.tfrec', batch_size=16, use_augmentation=True)
test_dataset = create_gcs_dataset('gs://tensorflow-colab-tpu/test_tfrec_files/*.tfrec', batch_size=16, use_augmentation=False)

# For training with separate validation:
# train_ds, val_ds = create_train_val_datasets('train_shard_*.tfrec', 'val_shard_*.tfrec')

# Test the pipeline
# Alternative version with side-by-side comparison

def test_pipeline(dataset, num_batches=1, num_samples_per_batch=16):
    """Test pipeline with side-by-side image and mask display"""
    print("Testing pipeline with side-by-side visualization...")

    for i, (images, masks) in enumerate(dataset.take(num_batches)):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images min/max: {tf.reduce_min(images):.3f} / {tf.reduce_max(images):.3f}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Masks dtype: {masks.dtype}")
        print(f"  Masks min/max: {tf.reduce_min(masks):.3f} / {tf.reduce_max(masks):.3f}")
        print()

        # Display images and masks side by side
        batch_size = min(images.shape[0], num_samples_per_batch)

        # Create subplot grid: 2 columns (image, mask) x batch_size rows
        fig, axes = plt.subplots(batch_size, 2, figsize=(8, 4 * batch_size))

        # Handle case when batch_size is 1
        if batch_size == 1:
            axes = axes.reshape(1, -1)

        for j in range(batch_size):
            # Convert tensors to numpy arrays
            img = images[j].numpy()
            mask = masks[j].numpy()

            # Display original image
            axes[j, 0].imshow(img)
            axes[j, 0].set_title(f'Image {j+1}')
            axes[j, 0].axis('off')

            # Display mask
            mask_display = np.squeeze(mask)
            axes[j, 1].imshow(mask_display, cmap='gray')
            axes[j, 1].set_title(f'Mask {j+1}')
            axes[j, 1].axis('off')

        plt.suptitle(f'Batch {i+1} - Images and Masks', fontsize=16)
        plt.tight_layout()
        plt.show()

        print(f"  Displayed {batch_size} samples from batch {i+1}")
        print(f"  Unique mask values in batch: {np.unique(masks.numpy())}")
        print("-" * 50)

# Example test:
# dataset = create_local_dataset('shard_*.tfrec', batch_size=4)
test_pipeline(train_dataset)