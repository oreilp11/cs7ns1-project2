import os
import cv2
import numpy
import argparse
import tflite_runtime.interpreter as tflite
from utils import time_func


#def decode(characters, labelmap, y):
#    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
#    return ''.join([labelmap[characters[x]] for x in y])


def classify(model, img):
    i = model.get_input_details()[0]['index']
    input = model.get_tensor(i)[0]
    input[:,:] = img
    model.invoke()
    outdict = model.get_output_details()[0]
    output = numpy.squeeze(model.get_tensor(outdict['index']))
    output = outdict['quantization'][0]*(output - outdict['quantization'][1])
    maxval = numpy.argpartition(-output, 1)
    return [(i, output[i]) for i in maxval[:1]][0]


@time_func
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--model-path', help='Model name to use for classification', type=str, required=True)
    parser.add_argument('-c','--captcha-dir', help='Where to read the captchas to break', type=str, required=True)
    parser.add_argument('-o','--output', help='File where the classifications should be saved', type=str, required=True)
    parser.add_argument('-s','--symbols', help='File with the symbols to use in captchas', type=str, required=True)
    parser.add_argument('-l','--labels', help='File with the labels to use in captchas', type=str, required=True)
    args = parser.parse_args()

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()
    with open(args.labels, 'r') as labels_file:
        captcha_labels = labels_file.readline().strip()
    symbols_dict = {label:symbol for symbol, label in zip(captcha_symbols,captcha_labels)}
    print(f"Classifying captchas with symbol set {captcha_symbols} and labels {captcha_labels}")


    with open(args.output, 'w') as output_file:
        model = tflite.Interpreter(args.model_path)
        print("Model loaded")
        model.allocate_tensors()
        print(model.get_input_details())

        for x in os.listdir(args.captcha_dir):
            # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x), 0)
            image = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            h, w = image.shape
            image = image.reshape([-1, h, w, 1])
            prediction, prob = classify(model, image)
            output_file.write(f'{x}, {symbols_dict[captcha_symbols[prediction]]}\n')

            print(f'Classified {x}')


if __name__ == '__main__':
    main()
