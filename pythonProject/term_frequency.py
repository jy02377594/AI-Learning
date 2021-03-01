import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

df = pd.read_csv('Total.csv')
stopwords = ['@', '-', '_','—', '','i',"i'm", '&',"it","la","the","de","en","que","this","el","floyd", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
                   "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
                   "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
                   "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
                   "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
                   "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
                   "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
                   "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
                   "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
                   "you've", "your", "yours", "yourself", "yourselves", "will", "rt", "---", "→", "℃", "-", ',',",", "，","go", "back", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

tracks = df[['track_en', 'other_info_en']]

terms = dict()
for index, row in tracks.iterrows():
    track = str(row["track_en"]) + str(row["other_info_en"])
    track_terms = track.split(" ")
    for term in track_terms:
#stemming
        term = term.replace(",","")
        term = term.replace(")","")
        term = term.replace(".","")
        term = term.replace("(","")

        if term in stopwords:
            continue
        if term in terms:
            terms[term] += 1
        else:
            terms[term] = 1



# extract high frequency terms
min_threshold=1000
frequency_list = terms.keys()
results = []
for word in frequency_list:
    if min_threshold is not None:
        if terms[word] < min_threshold:
            continue
    tuple = (word, terms[word])
    results.append(tuple)

byFreq = sorted(results, key=lambda word: word[1], reverse=True)
print("terms:", len(byFreq))

## plot frequent terms using 20 top words
sorted_wfreq = byFreq[0:20]
final_wfreq = dict()
for word, freq in sorted_wfreq:
    final_wfreq[word] = freq

sorted_wfreq = sorted(final_wfreq.items(), key=operator.itemgetter(1))
words_names = []
words_count = []

for word, freq in sorted_wfreq:
    words_names.append(word)
    words_count.append(freq)

show_plot = True
if show_plot == True:
    #
    fig, ax = plt.subplots()
    width = 0.56 # the width of the bars
    ind = np.arange(len(words_count))  # the x locations for the groups
    ax.barh(ind, words_count, width, color="#3366cc")
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(words_names, minor=False)
    plt.title('Word Frequency')
    plt.xlabel('Frequencies')
    plt.ylabel('Words')
    for i, v in enumerate(words_count):
        ax.text(v + 0.2, i - .15, str(v), color='black', fontweight='bold')
    plt.show()
