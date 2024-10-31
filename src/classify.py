import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import numpy
import argparse
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("-"*30)
        print(f"Runtime: {end-start:0.2f}s")
    return wrapper


def classify(model, img):
    model.set_tensor(model.get_input_details()[0]['index'], img)
    model.invoke()
    preds = model.get_tensor(model.get_output_details()[0]['index'])
    return numpy.argmax(numpy.array(preds), axis=1)[0]


@time_func
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--model-name', help='Model name to use for classification', type=str, required=True)
    parser.add_argument('-c','--captcha-dir', help='Where to read the captchas to break', type=str, required=True)
    parser.add_argument('-o','--output', help='File where the classifications should be saved', type=str, required=True)
    parser.add_argument('-s','--symbols', help='File with the symbols to use in captchas', type=str, required=True)
    parser.add_argument('-l','--labels', help='File with the labels to use in captchas', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()
    with open(args.labels, 'r') as labels_file:
        captcha_labels = labels_file.readline().strip()
    labels_dict = {label:symbol for symbol, label in zip(captcha_symbols, captcha_labels)}
    print(f"Classifying captchas with symbol set {captcha_symbols} and labels {captcha_labels}")

    with tf.device('/cpu:0'):
        with open(args.output, 'w') as output_file:
            with open(args.model_name+'.json', 'r') as json_file:
                model = keras.models.model_from_json(json_file.read())
            model.load_weights(args.model_name+'.h5')
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['sparse_categorical_accuracy'])

            for captcha in tqdm(os.listdir(args.captcha_dir)):
                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(args.captcha_dir, captcha), 0)
                h, w = raw_data.shape
                raw_data = raw_data.reshape([-1, h, w, 1])
                prediction = model.predict(raw_data)
                prediction = numpy.argmax(numpy.array(prediction)[0])[0]
                output_file.write(f'{captcha}, {labels_dict[captcha_labels[prediction]]}\n')

            # model = tf.lite.Interpreter(args.model_name)
            # model.allocate_tensors()

            # for captcha in tqdm(os.listdir(args.captcha_dir)):
            #     # load image and preprocess it
            #     raw_data = cv2.imread(os.path.join(args.captcha_dir, captcha), 0)
            #     h, w = raw_data.shape
            #     raw_data = raw_data.reshape([-1, h, w, 1])
            #     prediction = classify(model, raw_data)
            #     print(prediction)
            #     output_file.write(f'{captcha}, {labels_dict[captcha_labels[prediction]]}\n')

if __name__ == '__main__':
    main()
