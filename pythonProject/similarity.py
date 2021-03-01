import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random


df = pd.read_csv('Total.csv')
documents = df['track_en']

tracks = []
for track in df['track_en']:
    track = bytes(track, 'utf-8').decode()
    tracks.append(track)

raw_documents = random.sample(tracks, 20)

def create_binary_vector(raw_attr, entire_attributes):
    vec1 = []
    for t in entire_attributes:
        if t in raw_attr:
            vec1.append(1)
        else:
            vec1.append(0)
    return vec1


def consine_similarity(raw1, raw2):
    attribute_list = []
    raw1_attrs = raw1.split()
    raw2_attrs = raw2.split()

    # form attribute list
    for t in (raw1_attrs + raw2_attrs):
        if t not in attribute_list:
            attribute_list.append(t)

    # form vector for similarity
    vec1 = create_binary_vector(raw1_attrs, attribute_list)
    vec2 = create_binary_vector(raw2_attrs, attribute_list)

    sim = cosine_similarity([vec1], [vec2])
    sim1 = sim[0]

    return sim1[0]


def compute_pair_similarities(raw_documents):
    index1 = 0
    doc_count = len(raw_documents)
    sims = np.zeros((doc_count, doc_count))
    for raw1 in raw_documents:
        index2 = 0
        for raw2 in raw_documents:
            sim = consine_similarity(raw1, raw2)
            sims[index1, index2] = sim

            index2 = index2 + 1

        index1 = index1 + 1
    return sims


sims = compute_pair_similarities(raw_documents)
#print(sims)
ax = sns.heatmap(sims, linewidth=0.01)
plt.show()