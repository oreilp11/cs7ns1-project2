import requests
import time
import argparse
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor


def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("-"*30)
        print(f"Runtime: {end-start:0.2f}s")
    return wrapper


def check_missing_files(args):
    completed_imgs = set(os.listdir(args.img_path))
    with open(args.file_path) as file:
        all_imgs = {line.strip() for line in file}

    remaining_imgs = sorted(all_imgs - completed_imgs)
    with open(args.missed_path, "w") as file:
        for img in remaining_imgs:
            file.write(f"{img}\n")

    return remaining_imgs


def get_first_file(args):
    payload = {'shortname': args.shortname}
    with requests.get(args.url, stream=True, params=payload) as r:
        if r.status_code == 200:
            with open(args.file_path, "wb") as f:
                for content in r.iter_content():
                    f.write(content)
        print(r.status_code)

        return r.status_code == 200


def get_remaining_images(args):

    remaining_files = check_missing_files(args.img_path, args.file_path, args.missed_path)
    while len(remaining_files) > 0:
        print(f"Files remaining: {len(remaining_files)}")
        try:
            # for i, file in enumerate(remaining_files):    
            #     download_file(file, i, args)
            with ThreadPoolExecutor(args.max_workers) as tpex:
                for i, file in enumerate(remaining_files):
                    tpex.submit(download_file, file, i, args)
        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            print("Keyboard interrupt - checking remaining files:")
            check_missing_files(args.img_path, args.file_path, args.missed_path)
            break
        finally:
            remaining_files = check_missing_files(args.img_path, args.file_path, args.missed_path)
            print("-"*30)


def download_file(file, filenum, args):
    payload = {'shortname':args.shortname, 'myfilename':file}
    with requests.get(args.url, stream=True, params=payload) as r:
        if r.status_code == 200:
            with open(os.path.join(args.img_path, file), "wb") as f:
                for chunk in r.iter_content():
                    f.write(chunk)
        print(f"[{filenum}] {r.status_code} - {file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download files with parallel requests.")
    parser.add_argument("-n","--max_workers", type=int, default=5, help="Maximum number of parallel workers.", required=False)
    parser.add_argument("-f","--file-path", type=str, help="Maximum number of parallel workers.", required=True)
    parser.add_argument("-i","--img-path", type=str, help="Maximum number of parallel workers.", required=True)
    parser.add_argument("-m","--missed-path", type=str, help="Maximum number of parallel workers.", required=True)
    parser.add_argument("-s","--shortname", type=str, help="Maximum number of parallel workers.", required=True)
    parser.add_argument("-u","--url", type=str, help="Maximum number of parallel workers.", required=True)
    args = parser.parse_args()
    return args


@time_func
def main():
    args = parse_args()
    if not os.path.exists(args.file_path):
        with open(args.file_path,'w'):
            pass
    if not os.path.exists(args.img_path):
        os.makedirs(args.img_path)
    if not os.path.exists(args.missed_path):
        with open(args.missed_path,'w'):
            pass
    
    payload = {"shortname": args.shortname}

    if get_first_file(args.url, payload, args.file_path):
        get_remaining_images(args)

if __name__ == "__main__":
    main()