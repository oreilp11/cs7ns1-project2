import os
import cv2
import numpy as np
import argparse
import tflite_runtime.interpreter as tflite
from tqdm import tqdm
from utils import time_func


@time_func
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--model-path', help='Name of the modedl to be used in classification on PI', type=str, required=True)
    parser.add_argument('-c','--captcha-dir', help='Path to the directory of the captchas stored to be classified on PI', type=str, required=True)
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
    labels_dict = {label:symbol for symbol, label in zip(captcha_symbols,captcha_labels)}
    print(f"Classifying captchas with symbol set {captcha_symbols} and labels {captcha_labels}")

    with open(args.output, 'w') as output_file:
        model = tflite.Interpreter(args.model_path)
        model.allocate_tensors()
        for captcha in tqdm(os.listdir(args.captcha_dir)):
            raw_data = cv2.imread(os.path.join(args.captcha_dir, captcha), 0)
            h, w = raw_data.shape
            raw_data = raw_data.reshape([-1, h, w, 1]).astype(np.float32)
            model.set_tensor(model.get_input_details()[0]['index'], raw_data)
            model.invoke()
            prediction = np.array(model.get_tensor(model.get_output_details()[0]['index']))
            prediction = np.argmax(prediction, axis=1)[0]
            output_file.write(f'{captcha}, {labels_dict[captcha_labels[prediction]]}\n')


if __name__ == '__main__':
    main()
