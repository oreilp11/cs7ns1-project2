import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import argparse
from tqdm import tqdm
import tflite_runtime as tflite
from utils import clean_img, split_img, center_pad, time_func

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model-path', help='Name of the modedl to be used in classification on PI', type=str, required=True)
    parser.add_argument('-c','--captcha-dir', help='Path to the directory of the captchas stored to be classified on PI', type=str, required=True)
    parser.add_argument('-o','--output', help='File where the classifications should be saved', type=str, required=True)
    parser.add_argument('-s','--symbols', help='File with the symbols to use in captchas', type=str, required=True)
    parser.add_argument('-l','--labels', help='File with the labels to use in captchas', type=str, required=True)
    parser.add_argument('-n','--shortname', help='Shortname for csv submission', type=str, required=True)
    args = parser.parse_args()
    return args


@time_func
def main():
    args = parse_args()

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()
    with open(args.labels, 'r') as labels_file:
        captcha_labels = labels_file.readline().strip()
    labels_dict = {label:symbol for symbol, label in zip(captcha_symbols,captcha_labels)}
    print(f"Classifying captchas with symbol set {captcha_symbols} and labels {captcha_labels}")

    results = []
    model = tflite.Interpreter(args.model_path)
    model.allocate_tensors()
    for captcha in tqdm(os.listdir(args.captcha_dir)):
        raw_data = clean_img(cv2.imread(os.path.join(args.captcha_dir, captcha), 0))
        arr = []
        chars = [*split_img(raw_data)]
        for char in chars:
            char = center_pad(char)
            h, w = char.shape
            char = char.reshape([-1, h, w, 1]).astype(np.float32)
            model.set_tensor(model.get_input_details()[0]['index'], char)
            model.invoke()
            prediction = np.array(model.get_tensor(model.get_output_details()[0]['index']))
            prediction = np.argmax(prediction, axis=1)[0]
            arr.append(labels_dict[captcha_labels[prediction]])
        results.append((captcha,''.join(arr)))
    results.sort(key = lambda x:x[0])
    with open(args.output, 'w') as output_file:
        output_file.write(f'{args.shortname}\n')
        for captcha, label in results:
            output_file.write(f'{captcha},{label}\n')


if __name__ == '__main__':
    main()
