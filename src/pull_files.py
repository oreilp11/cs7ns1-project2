import requests
import time
import os


def check_missing_files(img_path, file_path, missed_path):
    done = set(os.listdir(img_path))
    with open(file_path) as file:
        total = {line.strip() for line in file}

    remaining = sorted(total - done)
    with open(missed_path, "w") as file:
        for r in remaining:
            file.write(f"{r}\n")


def get_first_file(url, payload, file_path):
    with requests.get(url, stream=True, params=payload) as r:
        if r.status_code == 200:
            with open(file_path, "wb") as f:
                for i in r.iter_content():
                    f.write(i)
        else:
            print(r.status_code)

        return r.status_code == 200


def get_remaining_images(url, payload, file_path, img_path, missed_path):
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    numfilesleft = 1  # temp value greater than zero
    while numfilesleft > 0:
        with open(file_path) as f:
            arr = [line.strip() for line in f]
        missed_files = []
        start, num = 0, 10
        try:
            while start < len(arr):
                for i, file in enumerate(arr[start : start + num]):
                    payload["myfilename"] = file
                    with requests.get(url, stream=True, params=payload) as r:
                        if r.status_code == 200:
                            with open(f"{img_path}/{file}", "wb") as f:
                                for i in r.iter_content():
                                    f.write(i)
                        else:
                            missed_files.append(file)
                            print(f"[{start+i}] {r.status_code} - {file}")
                            time.sleep(0.5)
                    time.sleep(0.5)

                start += num
                time.sleep(1.5)
        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            pass
        finally:
            numfilesleft = len(missed_files)
            if numfilesleft > 0:
                with open(missed_path, "w") as f:
                    for file in missed_files:
                        f.write(f"{file}\n")
            file_path = missed_path


if __name__ == "__main__":
    url = "https://cs7ns1.scss.tcd.ie"
    payload = {"shortname": "oreilp11"}
    file_path = os.path.join(os.path.dirname(__file__), "test.csv")
    img_path = os.path.join(os.path.dirname(__file__), "images")
    missed_path = os.path.join(os.path.dirname(__file__), "missed.csv")

    # if get_first_file(url, payload, file_path):
    #     get_remaining_images(url, payload, file_path, img_path, missed_path)

    check_missing_files(img_path, file_path, missed_path)
