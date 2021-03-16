import nltk
import numpy as np


def preprocess_sent(abstract, selected_bigrams, selected_unigrams):
    def content_feature(sent):
        sent_tokens = nltk.word_tokenize(sent)
        sent_unigrams = list(sent_tokens)
        sent_bigrams = list(nltk.bigrams(sent_tokens))

        temp_bigram = [sent_bigrams.count(bigram) for bigram in selected_bigrams]
        temp_unigram = [sent_unigrams.count(unigram) for unigram in selected_unigrams]
        feature = temp_bigram + temp_unigram

        return feature

    def location_feature(abstract):
        d_size = len(abstract) / 5

        feature = []
        division = 0
        for i in range(1, len(abstract)+1):
            if i > d_size:
                d_size = d_size + len(abstract)/5
                division = division+1
            feature.append(division)

        feature = np.array([feature]).transpose()
        return feature

    c_feature = []
    for sent in abstract:
        c_feature.append(content_feature(sent))
    c_feature = np.array(c_feature)
    l_feature = location_feature(abstract)

    # Merge features
    features = np.concatenate((c_feature, l_feature), axis=1)

    return features


def postprocess(int_labels):
    ans = []
    for label in int_labels:
        if label == 0:
            ans.append('BACKGROUND')
        elif label == 1:
            ans.append('OBJECTIVE')
        elif label == 2:
            ans.append('METHODS')
        elif label == 3:
            ans.append('RESULTS')
        elif label == 4:
            ans.append('CONCLUSIONS')
    return ans
