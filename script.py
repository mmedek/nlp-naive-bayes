#%% open and load data

import pandas as pd
import numpy as np
import os

def import_data(file):
    df = pd.read_csv(file, encoding="ISO-8859-1")
    return df

DATA_DIR = os.getcwd()
DATA_PATH = '\\data.csv'
full_path = DATA_DIR + DATA_PATH

df = import_data(full_path)
# there are loaded 3 empty columns so we want to delete them
df = df.drop(df.columns[2:5], axis=1)

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2)


#%% Naive bayes implementation
from collections import defaultdict

# create bag of words model for all words in mails
def get_counts_of_words(df):
    
    spam_words = defaultdict(int)
    ham_words = defaultdict(int)
    ham_count_words = 0
    spam_count_words = 0
    
    len_mails = len(df)
    
    for i in range(0, len_mails):
        
        splitted_words = df.iloc[i]['v2'].split()
        curr_class = df.iloc[i]['v1']
        
        for j in range(0, len(splitted_words)):
            
            if curr_class == 'spam':
                spam_count_words += 1
                spam_words[splitted_words[j]] += 1
            else:
                ham_count_words += 1
                ham_words[splitted_words[j]] += 1
                
    return [ham_words, spam_words, ham_count_words, spam_count_words]

# count of words in spams and hams for computing dependent prob
[spam_words, ham_words, ham_count_words, spam_count_words] = get_counts_of_words(train_df)
count_all_words = ham_count_words + spam_count_words

# prob that word is spam
prob_spam = spam_count_words / float(count_all_words)

# prob that word is NOT spam
prob_not_spam =  ham_count_words / float(count_all_words)

# alpha is value for Laplace smoothing - for unseen samples
def compute_prob(mail, spam, alpha = 1):
    
    prob = 1
    splitted_mail = mail.split()
    mail_len = len(splitted_mail)
    
    for i in range(0, mail_len):
        if spam:
            prob *= (spam_words[splitted_mail[i]] + alpha) / ((spam_count_words + alpha) * count_all_words)
        else:
            prob *= (ham_words[splitted_mail[i]] + alpha) / ((ham_count_words + alpha) * count_all_words)
    
    return prob
    
def classify(mail):
    
    isSpam = prob_spam * compute_prob(text, True)
    notSpam = prob_not_spam * compute_prob(text, False)
    
    spam = False
    prob = notSpam
    if isSpam > notSpam:
        spam = True
        prob = isSpam
    
    return [spam, prob]
    
text = test_df.iloc[0]['v2']
[isSpam, prob] = classify(text)

result = 'Text [' + text + '] is classified as '
if isSpam:
    result += 'SPAM '
else:
    result += 'HAM '

print(result)