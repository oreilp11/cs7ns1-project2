import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse
from captcha.image import ImageCaptcha
from utils import time_func, center_pad, clean_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--width', help='Width of captcha image', type=int, required=True)
    parser.add_argument('-H','--height', help='Height of captcha image', type=int, required=True)
    parser.add_argument('-c','--count', help='No of Captchas images to be generated for the sake of training', type=int, required=True)
    parser.add_argument('-o','--output-dir', help='Path to the directory where the generated captchas will be stored', type=str, required=True)
    parser.add_argument('-s','--symbols', help='path to file containing the symbols used in generation', type=str, required=True)
    parser.add_argument('-l','--labels', help='path to the file containing labels for sybmols', type=str, required=True)
    parser.add_argument('-f','--font', nargs="+", help='path to the file containing the font to use in captcha generation.', required=True)
    args = parser.parse_args()
    return args


def generate_captchas(args):
    captcha_generator = ImageCaptcha(width=args.width, height=args.height, fonts=[*args.font])

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()
    with open(args.labels, 'r') as labels_file:
        captcha_labels = labels_file.readline().strip()
    
    symbols_dict = {symbol:label for symbol, label in zip(captcha_symbols,captcha_labels)}

    print(f"Generating captchas with symbol set {captcha_symbols} and labels {captcha_labels}")

    os.makedirs(args.output_dir, exist_ok=True)
    for label in captcha_labels:
        os.makedirs(os.path.join(args.output_dir, label), exist_ok=True)
    
    n = len(captcha_symbols)
    for i in tqdm(range(args.count)):
        random_symbol = captcha_symbols[i % n]
        random_label = symbols_dict[random_symbol]

        image_path = os.path.join(args.output_dir, random_label, f'{random_label}.png')
        version = 1
        if os.path.exists(image_path):
            image_path = os.path.join(args.output_dir, random_label, f'{random_label}_{version}.png')
        while os.path.exists(image_path):
            version = max([int(v[:-4].split('_')[-1]) for v in os.listdir(os.path.join(args.output_dir, random_label)) if '_' in v]) + 1
            image_path = os.path.join(args.output_dir, random_label, f'{random_label}_{version}.png')
        
        image = captcha_generator.generate_image(random_symbol)
        image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2GRAY)
        image = center_pad(clean_img(image))
        cv2.imwrite(image_path, image)


@time_func
def main():
    args = parse_args()
    generate_captchas(args)
    

if __name__ == '__main__':
    main()
