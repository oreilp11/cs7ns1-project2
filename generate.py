#!/usr/bin/env python3

import os
import numpy
import random
import cv2
import argparse
import captcha.image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--width', help='Width of captcha image', type=int)
    parser.add_argument('-d','--height', help='Height of captcha image', type=int)
    parser.add_argument('-l','--length', help='Length of captchas in characters', type=int)
    parser.add_argument('-c','--count', help='How many captchas to generate', type=int)
    parser.add_argument('-o','--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('-s','--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    font = os.path.join(os.path.dirname(__file__),"Dreaming of Lilian.ttf")
    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=[font])

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    for _ in range(args.count):
        random_str = ''.join([random.choice(captcha_symbols) for j in range(args.length)])
        image_path = os.path.join(args.output_dir, random_str+'.png')
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')):
                version += 1
            image_path = os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')

        image = numpy.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)

if __name__ == '__main__':
    main()
