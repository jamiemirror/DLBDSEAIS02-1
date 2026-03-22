# Project
Source code for Sentiment analysis project.

# Purpose
The code is written for the analysis of the sentiment in customer reviews from Amazon marketplace. 
The marketing department decided to request automatic sentiment analysis tool that should be able to identify how customers feel about the products, based on their reviews and ratings.

# Data handling
The review must be given as csv data. The column with the text must be named "Text". The column with the scores must be named "score". The scores must be given as one to five (stars).

# main.py:
VADER model was selected due to already available vocabulary and sentiment analysis functionality.
The star numbers are transformed in the labels positive, negative or neutral.
The functionality displays first impression of the dataset and plots results. The subroutines in predef.py were used (dependency).

# predef.py:
Used for the analysis of text, based on the predefined VADER model.

# Dataset
The dataset for the analysis location: 
[Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

# Dependencies

1. The project was built, using Python 3.12.8. You need to download and install Python or if you have it already, you may check your version:

```sh
py --version #Windows OS
# OR
python3 --version # Linux / macOS
```

2. Run requirements installation:

```sh
pip install -r requirements.txt
```

# Run the code

To start the app locally, run from the project's directory: (default Debug mode is False)

```sh
py main.py       # Windows
# OR
python3 main.py  # macOS / Linux
```