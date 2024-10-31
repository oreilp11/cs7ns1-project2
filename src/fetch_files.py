import requests
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from utils import time_func

def check_missing_files(args):
    completed_imgs = set(os.listdir(args.img_path))
    with open(args.file_path) as file:
        all_imgs = {line.strip() for line in file}

    remaining_imgs = sorted(all_imgs - completed_imgs)
    with open(args.missed_path, "w") as file:
        for img in remaining_imgs:
            file.write(f"{img}\n")

    return remaining_imgs


def fetch_csv_file(args):
    payload = {'shortname': args.shortname}
    with requests.get(args.url, stream=True, params=payload) as r:
        if r.status_code == 200:
            with open(args.file_path, "wb") as f:
                for content in r.iter_content():
                    f.write(content)

        return r.status_code == 200


def fetch_remaining_files(args):
    remaining_files = check_missing_files(args)
    while len(remaining_files) > 0:
        print(f"Remaining files: {len(remaining_files)}")
        try:
            with ThreadPoolExecutor(args.max_workers) as tpex:
                for file in remaining_files:
                    tpex.submit(fetch_img_file, file, args)
        except KeyboardInterrupt:
            print("Keyboard interrupt - checking remaining files:")
            check_missing_files(args)
            break
        remaining_files = check_missing_files(args)
        print("-"*30)


def fetch_img_file(file, args):
    payload = {'shortname':args.shortname, 'myfilename':file}
    with requests.get(args.url, stream=True, params=payload) as r:
        if r.status_code == 200:
            with open(os.path.join(args.img_path, file), "wb") as f:
                for chunk in r.iter_content():
                    f.write(chunk)


def parse_args():
    parser = argparse.ArgumentParser(description="Download files with parallel requests.")
    parser.add_argument("-f","--file-path", type=str, help="path to list which contains the names of captchas to be downloaded from the url.", required=True)
    parser.add_argument("-i","--img-path", type=str, help="path to directory where the downloaded captchas will be stored", required=True)
    parser.add_argument("-s","--shortname", type=str, help="shortname of the user to access the resources", required=True)
    parser.add_argument("-u","--url", type=str, help="url of the site to request the resources", required=True)
    parser.add_argument("-m","--missed-path", type=str, help="path to file which keep a note of missed files", required=False, default=os.path.join(os.path.dirname(__file__),'temp.csv'))
    parser.add_argument("-n","--max_workers", type=int, default=4, help="Maximum number of parallel workers.", required=False)
    args = parser.parse_args()

    return args


@time_func
def main():
    args = parse_args()

    if not os.path.exists(args.file_path):
        os.makedirs(os.path.dirname(args.file_path), exist_ok=True)
        open(args.file_path,'w').close()

    if not os.path.exists(args.img_path):
        os.makedirs(args.img_path)

    if not os.path.exists(args.missed_path):
        os.makedirs(os.path.dirname(args.missed_path), exist_ok=True)
        open(args.missed_path,'w').close()

    if fetch_csv_file(args):
        fetch_remaining_files(args)
    
    os.remove(args.missed_path)

if __name__ == "__main__":
    main()