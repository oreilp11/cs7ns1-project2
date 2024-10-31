# Codebase for CS7NS1 Project 2
Completed by Paul O'Reilly and Duddupudi Daya Lokesh

## Process Description
### Pi
1. Configure venv and install necessary packages from `requirements-pi.txt` (Optional - build opencv from source for architecture specific optimisations)
2. Fetch initial csv and subsequent files from server using `fetch_files.py`
3. Classify the captcha set with a trained .tflite model using `rasp_classify.py`

### Training PC
1. Configure venv and install necessary packages from `requirements-pc.txt`
2. Fetch initial csv and subsequent files from server using `fetch_files.py`
3. Identify fonts/symbol set
4. Generate single character captcha training set using `generate_single.py`
5. Train cnn captcha solving model using `train_single.py`
6. (Optional) Classify the captcha set with a trained .tflite model using `classify.py`

## Instructions on how to run code
### Classifying on the Pi
The files you will need are `fetch_files.py` and `rasp_classify.py`. The prerequisite python packages are listed in the `requirements-pi.txt` file.
#### Fetching Files

    python .src/fetch_files.py -h
