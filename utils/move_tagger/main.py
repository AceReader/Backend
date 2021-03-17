"""Main program for training move tagger
Trained with Naive Bayes, SVM, CRF.

Required data:
    - abstracts.pickle

Output:
    - NB.joblib
    - SVM.joblib
    - CRF.joblib
    - selected_bigrams.txt
    - selected_unigrams.txt
"""

from datetime import datetime
import numpy as np

import sklearn_crfsuite
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, metrics
from joblib import dump

from preprocess_PUBMED import readData_PUBMED
from preprocess_PUBMED import content_features, location_feature
from process_input import preprocess_sent, postprocess

dataSize = 100
print("data size = ", dataSize)
# start_time = datetime.now()

# Read data
data = readData_PUBMED(data_size=dataSize)

# Feature selection
feature_bigrams, feature_unigrams, labels, selected_bigrams, selected_unigrams = content_features(data, top_k=dataSize)
feature_location = location_feature(data)

# Save selected features
with open('selected_bigrams.txt', 'w') as f:
    for b in selected_bigrams:
        f.write(' '.join(b))
        f.write('\n')

with open('selected_unigrams.txt', 'w') as f:
    for u in selected_unigrams:
        f.write(u)
        f.write('\n')

# Merging all fearures
features = np.concatenate((feature_bigrams, feature_unigrams, feature_location), axis=1)

# Split data
all_sents = len(features)
X_train, X_test = np.split(features, [int(all_sents*0.7)])
y_train, y_test = np.split(labels, [int(all_sents*0.7)])

# Prepare data for CRF model
# X_train
r0 = X_train.shape[0]
r1 = X_train.shape[1]

crf_List0 = []

xy = [str(x) for x in range(r1)]

for i in range(r0):
    xyz = list(X_train[i])
    X_crf = zip(xy, xyz)
    d = dict(X_crf)
    crf_List0.append(d)

X_train_crf = []
X_train_crf.append(crf_List0)

# X_test
r0 = X_test.shape[0]
r1 = X_test.shape[1]

crf_List1 = []

xy = [str(x) for x in range(r1)]

for i in range(r0):
    xyz = list(X_test[i])
    X_crf = zip(xy, xyz)
    d = dict(X_crf)
    crf_List1.append(d)

X_test_crf = []
X_test_crf.append(crf_List1)

# y_train
lablelist = []
ls = y_train.shape
inls = int(ls[0])
for i in range(inls):
    lablelist.append(str(y_train[i]))
y_train_crf = []
y_train_crf.append(lablelist)

# y_test
lablelist = []
ls = y_test.shape
inls = int(ls[0])
for i in range(inls):
    lablelist.append(str(y_test[i]))
y_test_crf = []
y_test_crf.append(lablelist)

# Training
# Train SVM
start_time = datetime.now()
clf = svm.SVC(kernel='linear')
# clf = svm.SVC(kernel='rbf',C=2,gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("SVM_training time : ", str(datetime.now()-start_time))
print("SVM_Accuracy:", metrics.accuracy_score(y_test, y_pred))
dump(clf, 'SVM.joblib')

# Train Naive Bayes
start_time = datetime.now()
mlt = MultinomialNB(alpha=1.0)
mlt.fit(X_train, y_train)
y_predict = mlt.predict(X_test)
print("NB_training time : ", str(datetime.now()-start_time))
print("NB_Accuracy：", metrics.accuracy_score(y_test, y_predict))
dump(mlt, 'NB.joblib')

# Train CRF
start_time = datetime.now()
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train_crf, y_train_crf)
y_pred_crf = crf.predict(X_test_crf)
print("CRF_training time : ", str(datetime.now()-start_time))
print("CRF_Accuracy：", metrics.accuracy_score(y_test_crf[0], y_pred_crf[0]))
dump(crf, 'CRF.joblib')

# Testing
# SVM_Testing
a = ['The importance of identifying rhetorical categories in texts has been widely acknowledged in the literature, since information regarding text organization or structure can be applied in a variety of scenarios, including genre-specific writing support and evaluation, both manually and automat- ically.', 'In this paper we present a Long Short-Term Memory (LSTM) encoder-decoder classifier for scientific abstracts.', 'As a large corpus of annotated abstracts was required to train our classifier, we built a corpus using abstracts extracted from PUBMED/MEDLINE.', 'Using the proposed classifier we achieved approximately 3% improvement in per-abstract ac- curacy over the baselines and 1% improvement for both per- sentence accuracy and f1-score.']
sent_feature = preprocess_sent(a, selected_bigrams, selected_unigrams)
sent_pred = clf.predict(sent_feature)
sent_pred = postprocess(sent_pred)
print("SVM predict : ", sent_pred)

# NB_Testing
a = ['The importance of identifying rhetorical categories in texts has been widely acknowledged in the literature, since information regarding text organization or structure can be applied in a variety of scenarios, including genre-specific writing support and evaluation, both manually and automat- ically.', 'In this paper we present a Long Short-Term Memory (LSTM) encoder-decoder classifier for scientific abstracts.', 'As a large corpus of annotated abstracts was required to train our classifier, we built a corpus using abstracts extracted from PUBMED/MEDLINE.', 'Using the proposed classifier we achieved approximately 3% improvement in per-abstract ac- curacy over the baselines and 1% improvement for both per- sentence accuracy and f1-score.']
sent_feature = preprocess_sent(a, selected_bigrams, selected_unigrams)
nb_sent_pred1 = mlt.predict(sent_feature)
nb_sent_pred = postprocess(nb_sent_pred1)
print("NB predict : ", nb_sent_pred)

# CRF_Testing
a = ['The importance of identifying rhetorical categories in texts has been widely acknowledged in the literature, since information regarding text organization or structure can be applied in a variety of scenarios, including genre-specific writing support and evaluation, both manually and automat- ically.', 'In this paper we present a Long Short-Term Memory (LSTM) encoder-decoder classifier for scientific abstracts.', 'As a large corpus of annotated abstracts was required to train our classifier, we built a corpus using abstracts extracted from PUBMED/MEDLINE.', 'Using the proposed classifier we achieved approximately 3% improvement in per-abstract ac- curacy over the baselines and 1% improvement for both per- sentence accuracy and f1-score.']
sent_feature = preprocess_sent(a, selected_bigrams, selected_unigrams)

# sent_feature_crf
r0 = sent_feature.shape[0]
r1 = sent_feature.shape[1]

crf_List0 = []

xy = [str(x) for x in range(r1)]

for i in range(r0):
    xyz = list(sent_feature[i])
    X_crf = zip(xy, xyz)
    d = dict(X_crf)
    crf_List0.append(d)

sent_feature_crf = []
sent_feature_crf.append(crf_List0)

crf_sent_pred1 = crf.predict(sent_feature_crf)
ar = []
for i in range(r0):
    ar.append(int(crf_sent_pred1[0][i]))

ar = np.array(ar)
crf_sent_pred = postprocess(ar)
print("CRF predict : ", crf_sent_pred)
