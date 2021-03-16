"""
feature engineering for SVM
"""
import pickle
import nltk
import numpy as np
from sklearn.feature_selection import SelectKBest,chi2

def readData_PUBMED(data_size=None):
    with open('abstracts.pickle','rb') as f:
        data = pickle.load(f)

    return data[:data_size]

def get_all_grams(grams, cur_all, cur_index):
    """
    for gram in grams:
        if gram not in cur_all.values():
            cur_all[cur_index] = gram
            cur_index += 1
    """

    for gram in list(set(grams)):
        if gram not in cur_all.keys():
            cur_all[gram] = cur_index
            cur_index += 1
    return cur_all, cur_index

def content_features(data, top_k):
    """
    Return top_k bigrams and unigrams from data
    """
    # Not necessary
    # Compare to selected features after chi-test
    #bigram_freqdist_BG = nltk.FreqDist()
    #bigram_freqdist_OBJ = nltk.FreqDist()
    #bigram_freqdist_MTD = nltk.FreqDist()
    #bigram_freqdist_RES = nltk.FreqDist()
    #bigram_freqdist_CON = nltk.FreqDist()

    all_bigrams = {}
    all_unigrams = {}
    index_bigram = 0
    index_unigram = 0

    label_matrix = []
    frequency_matrix_bigram = []
    frequency_matrix_unigram = []

    for abstract in data:
        for line in abstract:

            # Tokenize sentences
            tokens = nltk.word_tokenize(line[0])
            bigrams = nltk.bigrams(tokens)
            bigrams2 = nltk.bigrams(tokens) # ?
            cur_bigrams = list(nltk.bigrams(tokens))

            # Hash all bigrams and unigrams with grams as keys
            all_bigrams, index_bigram = get_all_grams(bigrams2, all_bigrams, index_bigram)
            all_unigrams, index_unigram = get_all_grams(tokens, all_unigrams, index_unigram)

            # Construct frequency_matrix
            temp_bigram = [ 0 for i in range(index_bigram) ]
            temp_unigram = [ 0 for i in range(index_unigram) ]
            for bigram in cur_bigrams:
                temp_bigram[ all_bigrams[bigram] ] += 1
            for unigram in tokens:
                temp_unigram[ all_unigrams[unigram] ] += 1
            #temp_bigram = [ 1 if bigram in cur_bigrams else 0 for bigram in all_bigrams.values() ]
            #temp_unigram = [ 1 if unigram in list(tokens) else 0 for unigram in all_unigrams.values() ]
            frequency_matrix_bigram.append(temp_bigram)
            frequency_matrix_unigram.append(temp_unigram)
            
            # Construct label_matrix
            if line[1]=='BACKGROUND':
                #bigram_freqdist_BG.update(bigrams)
                label_matrix.append(0)
            elif line[1]=='OBJECTIVE':
                #bigram_freqdist_OBJ.update(bigrams)
                label_matrix.append(1)
            elif line[1]=='METHODS':
                #bigram_freqdist_MTD.update(bigrams)
                label_matrix.append(2)
            elif line[1]=='RESULTS':
                #bigram_freqdist_RES.update(bigrams)
                label_matrix.append(3)
            elif line[1]=='CONCLUSIONS':
                #bigram_freqdist_CON.update(bigrams)
                label_matrix.append(4)
            else:
                pass
                #log.write('Not in any label {}\n'.format(line))

    #print(bigram_freqdist_BG.items())
    #print(bigram_freqdist_BG[('of','the')])

    #print(bigram_freqdist_BG.most_common(5))
    #print(bigram_freqdist_OBJ.most_common(5))
    #print(bigram_freqdist_MTD.most_common(5))
    #print(bigram_freqdist_RES.most_common(5))
    #print(bigram_freqdist_CON.most_common(5))

    # Pad to same length
    for i,line in enumerate(frequency_matrix_bigram):
        for j in range( len(all_bigrams)-len(line) ):
            frequency_matrix_bigram[i].append(0)
    for i,line in enumerate(frequency_matrix_unigram):
        for j in range( len(all_unigrams)-len(line) ):
            frequency_matrix_unigram[i].append(0)

    # Turn to array type
    frequency_matrix_bigram = np.array(frequency_matrix_bigram)
    frequency_matrix_unigram = np.array(frequency_matrix_unigram)
    label_matrix = np.array(label_matrix)        
    #print(frequency_matrix.shape) # lines * bigrams
    #print(label_matrix.shape)

    # Chi-test
    model_bigram = SelectKBest(chi2, k = top_k)
    model_bigram.fit_transform(frequency_matrix_bigram, label_matrix)
    #print(model1.scores_)
    model_unigram = SelectKBest(chi2, k = top_k)
    model_unigram.fit_transform(frequency_matrix_unigram, label_matrix)

    # Get selected grams
    selected_bigrams, bigrams_index = get_selected_grams(model_bigram, all_bigrams)
    selected_unigrams, unigrams_index = get_selected_grams(model_unigram, all_unigrams)
    #print(selected_bigrams)
    #print(selected_unigrams)

    # Get selected matrix
    selected_bigrams_freq = frequency_matrix_bigram[:, bigrams_index]
    selected_unigrams_freq = frequency_matrix_unigram[:, unigrams_index]

    return selected_bigrams_freq, selected_unigrams_freq, label_matrix, selected_bigrams, selected_unigrams

def get_selected_grams(model, gram_dict):
    selected_gram = []
    selected_index = []
    for i, feature in enumerate(model.get_support()):
        if feature == True:
            for gram,index in gram_dict.items(): # another hashing here will be better
                if index==i:
                    selected_gram.append(gram)
                    break
            selected_index.append(i)
    return selected_gram, selected_index
    
def location_feature(data):
    
    feature_location = []
    num_of_sent = 0
    for abstract in data:
        d_size = len(abstract) / 5

        temp = []
        division = 0
        for i in range(1,len(abstract)+1):
            if i > d_size:
                d_size = d_size + len(abstract)/5
                division = division+1
            temp.append(division)
            num_of_sent+=1

        feature_location.extend(temp)

    feature_location = np.array([feature_location])
    feature_location = feature_location.transpose()

    return feature_location

# Testing code
if __name__=='__main__':

    # Read data
    data = readData_PUBMED(data_size = 100)

    # Feature selection
    feature_bigrams, feature_unigrams, labels, selected_bigrams, selected_unigrams = content_features(data, top_k=50)
    feature_location = location_feature(data)

    # Merging all features
    features = np.concatenate( (feature_bigrams, feature_unigrams, feature_location), axis=1 )
    print(features[9])
    print(features.shape)

    
