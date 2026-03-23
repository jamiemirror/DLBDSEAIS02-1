from predef import *
from sklearn import metrics
import numpy as np
import pandas as pd
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#name of dataset
dataset = './Reviews.csv'
#https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
#kagglehub.dataset_download("snap/amazon-fine-food-reviews")
#dataset = ("/kaggle/snap/amazon-fine-food-reviews/Reviews.csv")
#column "Text" contains text to analyse 
#column "Score" contains ratings 
#we will limit the number of rows for analyse for performance reasons
number = 1000
#number of text row for example
no_ex = 5
#predef model vader
#1 => makes some plots to the vader results 
plt_vader_res = 1
#compound > pos => marked as 1 compound <  neg marked as -1, between 0.
vader_pos_limit = 0.3
vader_neg_limit = -0.3
#1 show 5 star with no_cc_neg most negative value and 1 star with no_cc_pos most positive.
cc = 1
no_cc_neg = 0
no_cc_pos = 0
#decision tree
crit = "e" #criterion == entropy "e", gini "g"
fi = 1 #1 for a first impression on the dataset
ba = 1 #1 for some basic analysis of the dataset

#data feed
#import data from csv ("/kaggle/snap/amazon-fine-food-reviews/Reviews.csv")
df = pd.read_csv(dataset, encoding = 'cp850')
example = df['Text'][no_ex]
#downsizing to 1000
print("original size: ", df.shape, "\n")
df = df.head(number)
print("downsized: ", df.shape, "\n")
#positive, neutral and negative votes, based on the stars rating
def create_sentiment(score):
    if score==1 or score==2:
        return -1 #negative sentiment
    elif score==4 or score==5:
        return 1 #positive sentiment
    else:
        return 0 #neutral sentiment
df['Sentiment'] = df['Score'].apply(create_sentiment)
#first impression of dataset
if fi == 1:
    print("Shape of dataset: ", df.shape, "\n")
    print("First rows of dataset: ", df.head(), "\n")
    print("columns of the dataset: ", df.columns)
    try:
        print("Values for the example: ", df.values[no_ex], "\n")
        print("Score for the example: ", df['Score'].values[no_ex], "\n")
    except RuntimeError:
        print("number", no_ex, " not valid for example", "\n")
    #analysis
    print(df['Score'].value_counts().sort_index())
    ax = df['Score'].value_counts().sort_index().\
        plot(kind='bar', title='Number of reviews for each star', figsize =(10,5))
    ax.set_xlabel('Review')
    plt.show()
if ba == 1:
    #analysis
    print(df['Sentiment'].value_counts().sort_index())
    ax = df['Sentiment'].value_counts().sort_index().\
        plot(kind='bar', title='Number Sentiments in reviews', figsize =(10,5))
    ax.set_xlabel('Review')
    plt.show()
#the predefined model VADER
    f,a, cf = vader_analysis(df, plt_vader_res, vader_pos_limit, vader_neg_limit, cc, no_cc_pos, no_cc_neg)
