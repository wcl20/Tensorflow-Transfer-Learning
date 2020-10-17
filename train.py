import argparse
import glob
import os
from core.datasets import SimpleDatasetLoader
from core.preprocessing import ResizeWithAspectRatio
from core.preprocessing import ToArray
from core.nn import TransferModel
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Path to dataset")
    args = parser.parse_args()

    preprocessors = [ResizeWithAspectRatio(224, 224), ToArray()]
    dataset_loader = SimpleDatasetLoader(preprocessors)

    # Load data
    img_paths = glob.glob(f"{args.dataset}/*/*.jpg")
    data, labels = dataset_loader.load(img_paths)
    # Standardize data
    data = data.astype("float") / 255.
    # Transform labels
    label_encoder = LabelBinarizer()
    labels = label_encoder.fit_transform(labels)
    print(f"[INFO] Data: {data.shape}. Labels: {labels.shape}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    # Data augmentation
    augmentation = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Base model
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    for layer in base_model.layers:
        layer.trainable = False
    model = TransferModel.build(base_model, 256, len(label_encoder.classes_))

    optimizer = RMSprop(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("[INFO] Training Head ...")
    model.fit(
        augmentation.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        validation_data=(X_test, y_test),
        epochs=25,
        verbose=1
    )

    # Evaluate model
    preds = model.predict(X_test, batch_size=32)
    report = classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=label_encoder.classes_)
    print(report)

    # Unfreeze part of the base model
    for layer in base_model.layers[15:]:
        layer.trainable = True
    optimizer = SGD(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("[INFO] Fine tuning model ...")
    model.fit(
        augmentation.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        validation_data=(X_test, y_test),
        epochs=100,
        verbose=1
    )

    # Evaluate model
    preds = model.predict(X_test, batch_size=32)
    report = classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=label_encoder.classes_)
    print(report)

    # Save model
    model.save("weights.hdf5")


if __name__ == '__main__':
    main()
