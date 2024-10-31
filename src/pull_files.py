# import requests
# import time
# import os


# def check_missing_files(img_path, file_path, missed_path):
#     done = set(os.listdir(img_path))
#     with open(file_path) as file:
#         total = {line.strip() for line in file}

#     remaining = sorted(total - done)
#     with open(missed_path, "w") as file:
#         for r in remaining:
#             file.write(f"{r}\n")


# def get_first_file(url, payload, file_path):
#     with requests.get(url, stream=True, params=payload) as r:
#         if r.status_code == 200:
#             with open(file_path, "wb") as f:
#                 for i in r.iter_content():
#                     f.write(i)
#         else:
#             print(r.status_code)

#         return r.status_code == 200


# def get_remaining_images(url, payload, file_path, img_path, missed_path):
#     if not os.path.exists(img_path):
#         os.mkdir(img_path)

#     numfilesleft = 1  # temp value greater than zero
#     while numfilesleft > 0:
#         with open(file_path) as f:
#             arr = [line.strip() for line in f]
#         missed_files = []
#         start, num = 0, 10
#         try:
#             while start < len(arr):
#                 for i, file in enumerate(arr[start : start + num]):
#                     payload["myfilename"] = file
#                     with requests.get(url, stream=True, params=payload) as r:
#                         if r.status_code == 200:
#                             with open(f"{img_path}/{file}", "wb") as f:
#                                 for i in r.iter_content():
#                                     f.write(i)
#                         else:
#                             missed_files.append(file)
#                             print(f"[{start+i}] {r.status_code} - {file}")
#                             time.sleep(0.5)
#                     time.sleep(0.5)

#                 start += num
#                 time.sleep(1.5)
#         except Exception as e:
#             print(e)
#         except KeyboardInterrupt:
#             pass
#         finally:
#             numfilesleft = len(missed_files)
#             if numfilesleft > 0:
#                 with open(missed_path, "w") as f:
#                     for file in missed_files:
#                         f.write(f"{file}\n")
#             file_path = missed_path
            

# if __name__ == "__main__":
#     url = "https://cs7ns1.scss.tcd.ie"
#     payload = {"shortname": "oreilp11"}
#     file_path = os.path.join(os.path.dirname(__file__), "test.csv")
#     img_path = os.path.join(os.path.dirname(__file__), "images")
#     missed_path = os.path.join(os.path.dirname(__file__), "missed.csv")

#     if get_first_file(url, payload, file_path):
#         get_remaining_images(url, payload, file_path, img_path, missed_path)

#     check_missing_files(img_path, file_path, missed_path)

import requests
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("-"*30)
        print(f"Runtime: {end-start:0.2f}s")
    return wrapper


def check_missing_files(img_path, file_path, missed_path):
    done = set(os.listdir(img_path))
    with open(file_path) as file:
        total = {line.strip() for line in file}

    remaining = sorted(total - done)
    with open(missed_path, "w") as file:
        for r in remaining:
            file.write(f"{r}\n")


def download_file(url, payload, file, img_path):
    payload["myfilename"] = file
    with requests.get(url, stream=True, params=payload) as r:
        if r.status_code == 200:
            with open(os.path.join(img_path, file), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return file, True
        else:
            print(f"Failed to download {file} with status code {r.status_code}")
            return file, False


def get_first_file(url, payload, file_path):
    with requests.get(url, stream=True, params=payload) as r:
        if r.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(r.status_code)

        return r.status_code == 200


def get_remaining_images(url, payload, file_path, img_path, missed_path, max_workers=5):
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    numfilesleft = 1  # temp value greater than zero
    while numfilesleft > 0:
        with open(file_path) as f:
            arr = [line.strip() for line in f]
        missed_files = []

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(download_file, url, payload.copy(), file, img_path): file
                    for file in arr
                }

                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        _, success = future.result()
                        if not success:
                            missed_files.append(file)
                    except Exception as e:
                        print(f"Exception occurred while downloading {file}: {e}")
                        missed_files.append(file)

        except KeyboardInterrupt:
            print("Download interrupted by user.")
            break
        finally:
            numfilesleft = len(missed_files)
            if numfilesleft > 0:
                with open(missed_path, "w") as f:
                    for file in missed_files:
                        f.write(f"{file}\n")
            file_path = missed_path


@time_func
def main():
    parser = argparse.ArgumentParser(description="Download files with parallel requests.")
    parser.add_argument("-n","--max_workers", type=int, default=5, help="Maximum number of parallel workers.")
    parser.add_argument("-f","--file-path", type=str, default=5, help="Maximum number of parallel workers.")
    parser.add_argument("-i","--img-path", type=str, default=5, help="Maximum number of parallel workers.")
    parser.add_argument("-m","--missed-path", type=str, default=5, help="Maximum number of parallel workers.")
    parser.add_argument("-s","--shortname", type=str, default=5, help="Maximum number of parallel workers.")
    parser.add_argument("-u","--url", type=str, default=5, help="Maximum number of parallel workers.")
    args = parser.parse_args()

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
        get_remaining_images(args.url, payload, args.file_path, args.img_path, args.missed_path, max_workers=args.max_workers)
    check_missing_files(args.img_path, args.file_path, args.missed_path)

if __name__ == "__main__":
    main()