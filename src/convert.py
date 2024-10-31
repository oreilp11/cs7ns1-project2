import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import tensorflow as tf
import tensorflow.keras as keras
from utils import time_func

@time_func
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--model-name', help='Name of the model to be used in classification', type=str, required=True)

    args = parser.parse_args()

    with open(args.model_name+'.json', 'r') as json_file:
        model = keras.models.model_from_json(json_file.read())
    model.load_weights(args.model_name+'.h5')
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        metrics=['sparse_categorical_accuracy']
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    lite_model = converter.convert()
    with open(f"{args.model_name}.tflite", "wb") as lite_file:
        lite_file.write(lite_model)

if __name__ == "__main__":
    main()