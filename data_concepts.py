# data_concepts.py
# Keras data pipeline for CAM + concept training.
# Uses an existing folder structure:
#   data/
#     train/<class_name>/*.jpg
#     validation/<class_name>/*.jpg
#
# No downloading, no zip, no Torch. Num classes is inferred from subfolders.

from __future__ import print_function

from pathlib import Path
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD  = np.array([0.229, 0.224, 0.225], dtype="float32")


def preprocess_input(img):
    """
    img: HxWx3 uint8 (0..255) -> float32 normalized by ImageNet mean/std.
    This must match the normalization used in train_factor_regularized_keras.
    """
    img = img.astype("float32") / 255.0
    img = (img - MEAN) / STD
    return img


def get_keras_generators(
    root="data",
    val_data = "validation",
    img_size=224,
    batch_size=64,
):
    """
    Returns:
        train_gen, val_gen, num_classes, class_names

    Expects the following directory structure:

        <root>/
          train/
            class_0/
            class_1/
            ...
          validation/
            class_0/
            class_1/
            ...

    Number of classes is automatically inferred from subfolders
    under train/ and validation/ via flow_from_directory.
    """
    root = Path(root)
    train_dir = root / "train"
    val_dir = root / val_data

    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(
            "[data_concepts] Expected folders:\n"
            "  {}/train\n"
            "  {}/validation_for_train\n"
            "But at least one of them does not exist.".format(root, root)
        )

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=10.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    # Only normalization for validation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="sparse",   # returns labels as integer indices
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
    )

    # Automatically infer class names and num_classes
    idx2class = dict((v, k) for k, v in train_gen.class_indices.items())
    num_classes = len(idx2class)
    class_names = [idx2class[i] for i in range(num_classes)]

    print("[data_concepts] Found {} classes: {}".format(num_classes, class_names))
    print("[data_concepts] Train samples:", train_gen.samples)
    print("[data_concepts] Val samples:", val_gen.samples)

    return train_gen, val_gen, num_classes, class_names
