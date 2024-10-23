#!/usr/bin/env python3

import os
import cv2
import time
import numpy
import argparse
import tflite_runtime.interpreter as tflite

def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("-"*30)
        print(f"Runtime: {end-start:0.2f}s")
    return wrapper

def decode(characters, labelmap, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([labelmap[characters[x]] for x in y])

@time_func
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--model-name', help='Model name to use for classification', type=str, required=True)
    parser.add_argument('-c','--captcha-dir', help='Where to read the captchas to break', type=str, required=True)
    parser.add_argument('-o','--output', help='File where the classifications should be saved', type=str, required=True)
    parser.add_argument('-s','--symbols', help='File with the symbols to use in captchas', type=str, required=True)
    parser.add_argument('-l','--labels', help='File with the labels to use in captchas', type=str, required=True)
    args = parser.parse_args()

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()
    with open(args.labels, 'r') as labels_file:
        captcha_labels = labels_file.readline().strip()
    symbols_dict = {symbol:label for symbol, label in zip(captcha_symbols,captcha_labels)}
    print(f"Classifying captchas with symbol set {captcha_symbols} and labels {captcha_labels}")

    with tf.device('/cpu:0'):
        with open(args.output, 'w') as output_file:
            with open(args.model_name+'.json', 'r') as json_file:
                model = keras.models.model_from_json(json_file.read())
            model.load_weights(args.model_name+'.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            for x in os.listdir(args.captcha_dir):
                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(rgb_data) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])
                prediction = model.predict(image)
                output_file.write(f'{x}, {decode(captcha_symbols, symbols_dict, prediction)}\n')

                print(f'Classified {x}')

if __name__ == '__main__':
    main()
