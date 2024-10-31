#!/usr/bin/env python3

import os
import numpy
import random
import time
from tqdm import tqdm
import cv2
import argparse
import captcha.image


def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("-"*30)
        print(f"Runtime: {end-start:0.2f}s")
    return wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--width', help='Width of captcha image', type=int, required=True)
    parser.add_argument('-H','--height', help='Height of captcha image', type=int, required=True)
    parser.add_argument('-l','--length', help='Length of captchas in characters', type=int, required=True)
    parser.add_argument('-c','--count', help='How many captchas to generate', type=int, required=True)
    parser.add_argument('-o','--output-dir', help='Where to store the generated captchas', type=str, required=True)
    parser.add_argument('-s','--symbols', help='File with the symbols to use in captchas', type=str, required=True)
    parser.add_argument('-L','--labels', help='File containing labels for sybmols', type=str, required=False)
    parser.add_argument('-f','--font', help='File with the font to use in captchas', type=str, required=True)
    args = parser.parse_args()
    return args


def generate_captchas(args):
    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=[args.font])

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip() 

    print(f"Generating captchas with symbol set {captcha_symbols}")

    if not os.path.exists(args.output_dir):
        print(f"Creating output directory {args.output_dir}")
        os.makedirs(args.output_dir)

    for _ in tqdm(range(args.count)):
        random_symbolstr = ''.join([random.choice(captcha_symbols) for _ in range(args.length)])
        image_path = os.path.join(args.output_dir, f'{random_symbolstr}.png')
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, f'{random_symbolstr}_{version}.png')):
                version += 1
            image_path = os.path.join(args.output_dir, f'{random_symbolstr}_{version}.png')

        image = numpy.array(captcha_generator.generate_image(random_symbolstr))
        cv2.imwrite(image_path, image)


def generate_captchas_with_labels(args):
    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=[args.font])

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()
    with open(args.labels, 'r') as labels_file:
        captcha_labels = labels_file.readline().strip()
    
    symbols_dict = {symbol:label for symbol, label in zip(captcha_symbols,captcha_labels)}
    #print(symbols_dict)
    print(f"Generating captchas with symbol set {captcha_symbols} and labels {captcha_labels}")

    if not os.path.exists(args.output_dir):
        print(f"Creating output directory {args.output_dir}")
        os.makedirs(args.output_dir)

    for _ in tqdm(range(args.count)):
        random_symbolstr = "".join([random.choice(captcha_symbols) for _ in range(args.length)])
        random_label = "".join([symbols_dict[random_symbol] for random_symbol in random_symbolstr])

        image_path = os.path.join(args.output_dir, f'{random_label}.png')
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, f'{random_label}_{version}.png')):
                version += 1
            image_path = os.path.join(args.output_dir, f'{random_label}_{version}.png')

        image = numpy.array(captcha_generator.generate_image(random_symbolstr))
        cv2.imwrite(image_path, image)

@time_func
def main():
    args = parse_args()

    if args.labels is None:
        generate_captchas(args)
    else:
        generate_captchas_with_labels(args)
    

if __name__ == '__main__':
    main()
