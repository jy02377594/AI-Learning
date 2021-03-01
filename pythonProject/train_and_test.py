import pandas as pd
from pandas.core.series import Series
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


# df_plane = pd.read_csv("Plane_data.csv", encoding='utf8')
# df_plane['label'] = Series(['PLANE']*len(df_plane), index=df_plane.index)
# train_plane, test_plane = train_test_split(df_plane, test_size=0.3)
#
# df_train = pd.read_csv("Train_data.csv", encoding='utf8')
# df_train['label'] = Series(['TRAIN']*len(df_train), index=df_train.index)
# train_train, test_train = train_test_split(df_train, test_size=0.3)
# #
# df_automobile = pd.read_csv("Automobile_data.csv", encoding='utf8')
# df_automobile['label'] = Series(['AUTOMOBILE']*len(df_automobile), index=df_automobile.index)
# train_automobile, test_automobile = train_test_split(df_automobile, test_size=0.3)

# plane training
# train = train_plane

# train,subway training
# train = train_train

# automobile training
# train = train_automobile

df_wuhan = pd.read_csv("WuhanRelated.csv", encoding='utf8')
df_wuhan['label'] = Series(['Wuhan']*len(df_wuhan), index=df_wuhan.index)
#train_wuhan, test_wuhan = train_test_split(df_wuhan, test_size=0.3)



df_nonwuhan = pd.read_csv("WuhanUnrelated.csv", encoding='utf8')
df_nonwuhan['label'] = Series(['nonWuhan']*len(df_nonwuhan), index=df_nonwuhan.index)
#train_nonwuhan, test_nonwuhan = train_test_split(df_nonwuhan, test_size=0.3)

# combine 3 classes together training
# train = pd.concat([train_automobile, train_train, train_plane], ignore_index=True, axis=0)
# test = pd.concat([test_automobile, test_train, test_plane], ignore_index=True, axis=0)

# wuhan related class training
#train = train_wuhan

df_total = pd.concat([df_wuhan, df_nonwuhan],ignore_index=True, axis=0, sort='False')
train, test = train_test_split(df_total, test_size=0.3)
# wuhan unrelated class training
# train = train_nonwuhan
#
# test = pd.concat([test_wuhan, test_nonwuhan], ignore_index=True, axis=0)

stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "will"]
#stop_words = stopwords.words('english')
extra_stopwords = ['@', '-', '_', '','i',"i'm", '&',"it","la","the","de","en","que","this","el","floyd"]
my_stopwords = stop_words + extra_stopwords
stemmer = SnowballStemmer("english")

start_with_stopwords = ["@", '#', 'http']


def is_stop_word(word):
    # if word in my_stopwords:
    #     return True

    # startswith test
    for st in start_with_stopwords:
        if word.startswith(st):
            return True

    return False


def pre_process_text(track):
    # words = [stemmer.stem(word.lower()) for word in track.split() if not is_stop_word(word.lower())]
    words = [word.lower() for word in track.split() if not is_stop_word(word.lower())]

    return ' '.join(words)


def clean_text_and_write_to_file(filename, df):

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["track_num", "track", "label"])

        for idx, row in df.iterrows():

            track_num = row["number"]
            track = row["track_en"]

            try:
                label = row["label"]
            except:
                print(filename, ", bad track ", id, "; track:", track)
                continue

            track = pre_process_text(track=track)
            if track.strip() == '':
                continue

            writer.writerow([track_num, track, label])

# training and testing files
clean_text_and_write_to_file('Wuhan_total_train_3.csv', train)
clean_text_and_write_to_file('Wuhan_total_test_3.csv', test)
