import os
import random
import argparse
from tqdm import tqdm
import time

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
    parser.add_argument('-i','--input_dir', help='How many captchas to generate', type=str, required=True)
    parser.add_argument('-o','--output-dir', help='Where to store the generated captchas', type=str, required=True)
    parser.add_argument('-s','--split', help='File with the symbols to use in captchas', type=int, required=False, default=0.2)
    args = parser.parse_args()
    return args


@time_func
def main():
    args = parse_args()
    imgs = sorted(os.listdir(args.input_dir))
    num_imgs = len(imgs)
    split_size = int(args.split*num_imgs)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for _ in tqdm(range(split_size)):
        file = random.choice(imgs)
        os.rename(os.path.join(args.input_dir,file), os.path.join(args.output_dir,file))
        imgs.remove(file)
        

if __name__ == '__main__':
    main()