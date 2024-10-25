import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from utils import time_func

def create_model(captcha_num_symbols, input_shape):
    model = keras.Sequential([
        keras.layers.Rescaling(scale=1./255,input_shape=input_shape),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.2),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer="he_uniform"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(captcha_num_symbols, activation='softmax')
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        metrics=["sparse_categorical_accuracy"]
    )

    model.summary()

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", help="Width of captcha image", type=int,required=True)
    parser.add_argument("-H", "--height", help="Height of captcha image", type=int,required=True)
    parser.add_argument("-b", "--batch-size", help="How many images in training captcha batches", type=int,required=True)
    parser.add_argument("-d", "--dataset-dir", help="Where to look for the training image dataset", type=str,required=True)
    parser.add_argument("-o", "--output-model-name", help="Where to save the trained model", type=str,required=True)
    parser.add_argument("-r", "--input-model",help="Where to look for the input model to continue training",type=str,required=False)
    parser.add_argument("-e", "--epochs", help="How many training epochs to run", type=int,required=True)
    parser.add_argument("-l", "--labels", help="File with the labels to use in captchas", type=str,required=True)
    args = parser.parse_args()
    return args


@time_func
def main():
    args = parse_args()

    if not os.path.exists(os.path.dirname(args.output_model_name)):
        os.makedirs(os.path.dirname(args.output_model_name))

    with open(args.labels) as labels_file:
        captcha_labels = labels_file.readline().strip()

    with tf.device("/cpu:0"):
        model = create_model(len(captcha_labels), (args.height, args.width, 1))

        training_data, validation_data = keras.preprocessing.image_dataset_from_directory(
            directory=args.dataset_dir,
            color_mode='grayscale',
            label_mode='int',
            class_names=captcha_labels,
            validation_split=0.2,
            subset="both",
            seed=2024,
            image_size=(args.height, args.width),
            batch_size=args.batch_size
        )

        callbacks = [
            keras.callbacks.EarlyStopping(patience=args.epochs // 10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(f'{args.output_model_name}.h5', save_best_only=True),
        ]

        # Save the model architecture to JSON
        with open(args.output_model_name + ".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit(
                x=training_data,
                validation_data=validation_data,
                epochs=args.epochs,
                callbacks=callbacks,
                use_multiprocessing=True,
            )
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt caught, saving current weights as {args.output_model_name}_resume.h5")
            model.save_weights(f"{args.output_model_name}_resume.h5")
        finally:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            lite_model = converter.convert()
            with open(f"{args.output_model_name}.tflite", "wb") as lite_file:
                lite_file.write(lite_model)


if __name__ == "__main__":
    main()