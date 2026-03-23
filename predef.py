import nltk
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('vader_lexicon')
# nltk.download('punkt')
import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import seaborn as sns
#for VADER
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

def results(y_test, preds, y, method):
    """calculate the f1 score and the accuracy"""    
    print("\n")
    print("Results for ", method, "\n")
    #confusion matrix
    conf = metrics.confusion_matrix(y_test, preds)
    print(conf, "\n")
    #f1 score
    f1 = metrics.f1_score(y_test, preds, average='weighted')
    #average =  micro, macro, weighted
    print("f1 = ", f1, "\n")
    #accuracy
    accuracy = metrics.accuracy_score(y_test, preds)
    print("accuracy = ", accuracy,  "\n")
    return f1, accuracy, conf

def vader_analysis(df, plt_vader_res, vader_pos_limit, vader_neg_limit, cc, no_cc_pos, no_cc_neg): 
    """lexicon approach VADER"""
    sia =  SentimentIntensityAnalyzer()
    #еxample
    #print(sia.polarity_scores("This is a good day"))
    #print(sia.polarity_scores("worst things ever"))
    #print(sia.polarity_scores(example))
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Text']
        myid = row['Id']
        res[myid] = sia.polarity_scores(text)
    vaders = pd.DataFrame(res)
    vaders = vaders.T
    vaders = vaders.reset_index().rename(columns={'index':'Id'})
    vaders = vaders.merge(df, how='left')
    #print(vaders)
    #VADER results plotting
    if plt_vader_res == 1:
        ax = sns.barplot(data=vaders, x='Score', y = 'compound')
        ax.set_title('Compound score by number of stars')
        plt.show()
        fix, axs = plt.subplots(1,3, figsize=(15,5))
        sns.barplot(data=vaders, x='Score', y = 'pos', ax = axs[0])
        sns.barplot(data=vaders, x='Score', y = 'neu', ax = axs[1])
        sns.barplot(data=vaders, x='Score', y = 'neg', ax = axs[2])
        axs[0].set_title('Positive')
        axs[1].set_title('Neutral')
        axs[2].set_title('Negative')
        plt.show()
    def estimate_sentiment(comp):
        """interpret the compound value as positive, negative or neutral, based on the pos/neg_vader_limit.: 
            compound > 0,3 as positive
            -0,3 < compound < 0,3 as neutral
            compound < - 0,3 as negative"""
        if comp >= vader_pos_limit:
            return 1 #positive sentiment estimated by text
        elif comp > vader_neg_limit:
            return 0 #neutral sentiment estimated by text
        else:
            return -1 #negative sentiment estimated by text
    vaders['Sentiment_est'] = vaders['compound'].apply(estimate_sentiment)
    #print(vaders.columns)
    #confusion_matrix
    y_test = vaders['Sentiment'].to_numpy()
    preds = vaders['Sentiment_est'].to_numpy()
    #calculate the f1 score and the accuracy
    f,a, cf = results(y_test, preds, y_test, "vader")
    return f,a, cf 
   