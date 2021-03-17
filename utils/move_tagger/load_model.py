"""
Load models
"""

from joblib import load
from process_input import preprocess_sent, postprocess


def model_load(model_name, selected_bigrams_name, selected_unigrams_name):
    """Load model

    Args:
        model_name (str): file name of the model.
        selected_bigrams_name (str): file name of bigram feature.
        selected_unigrams_name (str): file name of unigram feature.

    Returns:
        model: loaded model
        list: list of bigram feature.
        list: list of unigram feature.
    """

    model = load(model_name)

    selected_bigrams = []
    with open(selected_bigrams_name, 'r') as f:
        bigram_data = f.read()
    selected_bigrams = [b for b in bigram_data.split('\n')[:-1]]

    selected_unigrams = []
    with open(selected_unigrams_name, 'r') as f:
        unigram_data = f.read()
    selected_unigrams = [u for u in unigram_data.split('\n')[:-1]]

    return model, selected_bigrams, selected_unigrams


def model_predict(sent, model, selected_bigrams, selected_unigrams):
    """Predict with model

    Args:
        sent (str): Sentence to be classified.
        model (.joblib): the model used.
        selected_bigrams (list): list of bigram feature.
        selected_unigrams (list): list of unigram feature.

    Returns:
        str: output label.
    """

    sent_feature = preprocess_sent(sent, selected_bigrams, selected_unigrams)

    prediction = model.predict(sent_feature)
    prediction = postprocess(prediction)

    return prediction


if __name__ == "__main__":

    # Testing
    sent = ['The importance of identifying rhetorical categories in texts has \
            been widely acknowledged in the literature, since information \
            regarding text organization or structure can be applied in a \
            variety of scenarios, including genre-specific writing support \
            and evaluation, both manually and automatically.',
            'In this paper we present a Long Short-Term Memory (LSTM) \
            encoder-decoder classifier for scientific abstracts.',
            'As a large corpus of annotated abstracts was required to train \
            our classifier, we built a corpus using abstracts extracted from \
            PUBMED/MEDLINE.',
            'Using the proposed classifier we achieved approximately 3% \
            improvement in per-abstract ac- curacy over the baselines and 1% \
            improvement for both per- sentence accuracy and f1-score.']

    model_name = 'SVM.joblib'
    selected_bigrams_name = 'selected_bigrams.txt'
    selected_unigrams_name = 'selected_unigrams.txt'

    SVM, selected_bigrams, selected_unigrams = model_load(model_name, selected_bigrams_name, selected_unigrams_name)
    prediction = model_predict(sent, SVM, selected_bigrams, selected_unigrams)

    print(prediction)
