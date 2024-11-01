# Codebase for CS7NS1 Project 2
Completed by Paul O'Reilly _ID: 24351186_ and Duddupudi Daya Lokesh _ID: 24351819_
Group Number: 32

## Process Description
This codebase assumes that Python == 3.8 is installed on the Pi. It also assumes that the Training PC is a Windows machine and has Python >= 3.10 installed.

### Pi
1. Configure venv and install necessary packages from `requirements-pi.txt` 
2. _(Optional)_ Build OpenCV package from source for architecture specific optimisations
3. Fetch initial csv and subsequent files from server using `fetch_files.py`
4. Classify the captcha set with a trained .tflite model using `rasp_classify.py`

### Training PC
1. Configure venv and install necessary packages from `requirements-pc.txt`
2. Fetch initial csv and subsequent files from Pi or optionally from server using `fetch_files.py`
3. Identify fonts/symbol set
4. Generate single character captcha training set using `generate_single.py`
5. Train cnn captcha solving model using `train_single.py`
6. _(Optional)_ Classify the captcha set with a trained .tflite model using `classify.py`

## Instructions on How to Run Code
The below assumes you are in the root directory of the project i.e. parent directory of `src`

### Classifying on Pi
The files you will need are `fetch_files.py` and `rasp_classify.py`. The prerequisite python packages are listed in the `requirements-pi.txt` file.

#### Fetching Files
Run the below in a venv with the appropriate prerequisites installed:
```
python src/fetch_files.py -h
```

The `fetch_files.py` module has the following inputs:
- `-f / --file-path`: Path to save initial csv containing image file names from server.
- `-i / --img-path`: Path to save images listed in the initial csv.
- `-s / --shortname`: Shortname to use in the request sent to the server.
- `-u / --url`: URL to the server hosting the files.
- `-m / --missed-path`: _(Optional)_ Path for temporary csv used for calculating remaining files not yet downloaded. Defaults to the same directory as the `fetch_files.py` file.
- `-n / --max_workers`: _(Optional)_ Maximum number of workers for `ThreadPoolExecuter` to use for multiprocessing. Defaults to 4 due to number of cores available on Pi. 

Below is an example of how we used `fetch_files.py`:
```
python src/fetch_files.py -n 4 -f assets/oreilp11_fienames.csv -i assets/imgs/oreilp11 -s oreilp11 -u https://cs7ns1.scss.tcd.ie
```
#### Classifying Files
Run the below in a venv with the appropriate prerequisites installed and trained `.tflite` model copied to the pi:
```
python src/rasp_classify.py -h
```

The `rasp_classify.py` module has the following inputs:
- `-m / --model-path`: Path to trained `.tflite` model to use for classification.
- `-c / --captcha-dir`: Path to captchas to classify.
- `-s / --symbols`: Path to file containing guesses for captcha symbol set.
- `-l / --labels`: Path to file containing labels corresponding to guesses for captcha symbol set - used to ensure special characters do not interfere with string parsing/processing.
- `-o / --output`: Path to csv containing list of captchas and corresponding classififcations.
- `-n / --shortname`: Shortname to place at top of output csv in line with required format.

Below is an example of how we used `rasp_classify.py`:
```
python src/rasp_classify.py -m models/lilian-eamon/lilian-eamon.tflite -c assets/imgs/oreilp11-imgs -s assets/symbols.txt -l assets/labels.txt -o models/lilian-eamon/oreilp11.csv -n oreilp11
```

### Training Model on PC
The files you will need are `generate.py` and `train.py`. Optionally classification can be performed with `classify.py` The prerequisite python packages are listed in the `requirements-pc.txt` file. You will also require the files downloaded via `fetch_files.py`, these can be copied over from the Pi or otherwise can be fetched locally using `fetch_files.py`.

#### Generating Training Set
Run the below in a venv with the appropriate prerequisites installed:
```
python .\src\generate.py -h
```

The `generate.py` module has the following inputs:
- `-w / --width`: Width of generated training set captchas.
- `-H / --height`: Height of generated training set captchas.
- `-c / --count`: Number of training set captchas to generate.
- `-s / --symbols`: Path to file containing guesses for captcha symbol set.
- `-l / --labels`: Path to file containing labels corresponding to guesses for captcha symbol set - used to ensure special characters do not interfere with string parsing/processing.
- `-o / --output`: Path to directory to generate training set captchas in.
- `-f / --font`: Font(s) to use when generating training set. Minimum of 1 font to be used.

Below is an example of how we used `generate.py`:
```
python .\src\generate.py -w 96 -H 96 -c 120000 -s .\assets\symbols.txt -l .\assets\labels.txt -o .\models\lilian-eamon\training -f .\assets\fonts\DreamingOfLilian.ttf .\assets\fonts\Eamon.ttf
```

#### Training CNN model
Run the below in a venv with the appropriate prerequisites installed and training set generated:
```
python .\src\train.py -h
```

The `train.py` module has the following inputs:
- `-w / --width`: Width of generated training set captchas.
- `-H / --height`: Height of generated training set captchas.
- `-b / --batch-size`: Batch size to use when training. A value of 16 is recommended.
- `-e / --epochs`: Number of epochs to use when training. A value of 8 is recommended to help avoid overfitting.
- `-l / --labels`: Path to file containing labels corresponding to guesses for captcha symbol set - used to ensure special characters do not interfere with string parsing/processing.
- `-d / --dataset-dir`: Path to directory containing generated training set captchas.
- `-o / --output-model-name`: Path including name to save trained model (and weights file if training is paused).
- `-r / --resume-model`: _(Optional)_ Path to weights file (`.keras`) to resume training from if training was paused.

Below is an example of how we used `train.py`:
```
python .\src\train.py -w 96 -H 96 -b 16 -e 8 -l .\assets\labels.txt -d .\models\lilian-eamon\training -o .\models\lilian-eamon\lilian-eamon
```

#### (Optional) Classifying Files
Run the below in a venv with the appropriate prerequisites installed and trained `.tflite` model available locally:
```
python .\src\classify.py -h
```

The `classify.py` module has the following inputs:
- `-m / --model-path`: Path to trained `.tflite` model to use for classification.
- `-c / --captcha-dir`: Path to captchas to classify.
- `-s / --symbols`: Path to file containing guesses for captcha symbol set.
- `-l / --labels`: Path to file containing labels corresponding to guesses for captcha symbol set - used to ensure special characters do not interfere with string parsing/processing.
- `-o / --output`: Path to csv containing list of captchas and corresponding classififcations.
- `-n / --shortname`: Shortname to place at top of output csv in line with required format.

Below is an example of how we used `classify.py`:
```
python .\src\classify.py -m .\models\lilian-eamon\lilian-eamon.tflite -c .\assets\imgs\oreilp11-imgs -s .\assets\symbols.txt -l .\assets\labels.txt -o .\models\lilian-eamon\oreilp11_classified.csv -n oreilp11
```
