import json
import sys
from flask import Flask
from flask import request
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, './utils/grobid_client_python')
import grobid_client as grobid

sys.path.insert(1, './utils/parseXML')
from parseXML import parseXML

sys.path.insert(2, './utils/move_tagger')
from load_model import model_load, model_predict

app = Flask(__name__)
CORS(app)


@app.route('/api/upload_pdf', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        # Save upload file
        uploaded_file = request.files['uploaded_PDF']
        file_name = secure_filename(uploaded_file.filename)
        uploaded_file.save('./data/pdfInput/'+file_name)

        # Pdf to xml to json by grobid
        client = grobid.grobid_client(config_path="./utils/grobid_client_python/config.json")
        client.process("processFulltextDocument", "./data/pdfInput", output="./data/pdfOutput", n=20)
        content_dict = parseXML('./data/pdfOutput/', './data/jsonOutput/', file_name)

        # Move tagging
        p = './utils/move_tagger/'
        model_name = p+'SVM.joblib'
        selected_bigrams_name = p+'selected_bigrams.txt'
        selected_unigrams_name = p+'selected_unigrams.txt'
        SVM, selected_bigrams, selected_unigrams = model_load(model_name, selected_bigrams_name, selected_unigrams_name)
        sent = content_dict['abstract']
        prediction = model_predict(sent, SVM, selected_bigrams, selected_unigrams)
        content_dict['labels'] = prediction

        return {'status': '200', 'file_name': file_name, 'content_dict': content_dict}

    return 'nothing uploaded, get (400?)'


@app.route('/<file_name>', methods=['GET', 'POST'])
def content(file_name):
    with open('./data/jsonOutput/'+file_name+'.json') as f:
        content_dict = json.load(f)

    p = './utils/move_tagger/'
    model_name = p+'SVM.joblib'
    selected_bigrams_name = p+'selected_bigrams.txt'
    selected_unigrams_name = p+'selected_unigrams.txt'
    SVM, selected_bigrams, selected_unigrams = model_load(model_name, selected_bigrams_name, selected_unigrams_name)
    sent = content_dict['abstract'].split('.')
    prediction = model_predict(sent, SVM, selected_bigrams, selected_unigrams)

    content_dict['labels'] = prediction

    return content_dict
