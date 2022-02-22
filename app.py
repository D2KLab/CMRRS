"""

Flow:
    - Expose update_index and call it whenever there is a new content in the DB to add
    - Expose retrieve that returns sorted top_k given a query embedding

Doubts:
    - Faiss assigns an internal ID to each embedding in the `index` structure. We should build a mapping to the global ID of the content within the MV network
"""


from cgitb import text
from email.mime import image
import os
import io
import random
import string
from turtle import color
from typing import List, Tuple, Union
from datetime import datetime
from unicodedata import name

import faiss
import numpy as np
from flask import Flask, jsonify, request, render_template, flash, redirect
from logstash_formatter import LogstashFormatterV1

import torch
import clip
from PIL import Image
import base64

from werkzeug.utils import secure_filename

app            = Flask(__name__)
app.secret_key = "manbearpig_MUDMAN888"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
EMB_SIZE       = 512

app.config["IMAGE_UPLOADS"] = "C:\\Users\\fedde\\OneDrive\\Desktop\\mediaverse\\rest + clip + gui\\static\\images"

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


class ContentContainer:
    def __init__(self) -> None:
        ## Take trace of duplicates
        self.texts  = dict()
        self.images = dict()

    def check_duplicate(self, input: Union[str, np.array], content_id: str) -> bool:
        if isinstance(input, str):
            if input in set(self.texts.values()):
                flash('Text arleady present in the MV archive')
                return True
            self.texts[content_id] = input
        else:
            if input in set(self.images.values()):
                flash('Image arleady present in the MV archive')
                return True
            self.images[content_id] = input
        return False

    def get_text(self, contentid: str):
        return self.texts[contentid]

    def get_image(self, contentid: str):
        return self.images[contentid]

    def getMVarchive(self):
        for id, text in self.texts.items():
            flash('[MV] [TEXT] {}: {}'.format(id, text))
        for id, image in self.images.items():
            flash('[MV] [IMAGE] {}: {}'.format(id, image))


class ClipEncoder:
    def __init__(self) -> None:
        model, preprocess = clip.load("ViT-B/32") # (or load model starting from state_dict)
        self.model = model
        self.preprocess = preprocess

    def encode(self, input: Union[str, np.array]) -> np.array:
        if isinstance(input, str):
            text = clip.tokenize(input).to(DEVICE)
            return self.model.encode_text(text).detach().cpu().numpy().reshape(EMB_SIZE).astype(np.float32)
        else:
            image = self.preprocess(Image.open(input)).unsqueeze(0).to(DEVICE)
            return self.model.encode_image(image).detach().cpu().numpy().reshape(EMB_SIZE).astype(np.float32)


class Indexer:
    def __init__(self, emb_size: int=EMB_SIZE) -> None:
        # to get total length of flat index: index.xb.size()
        # to get number of embeddings in index: index.xb.size() // EMB_SIZE
        self.index        = faiss.index_factory(emb_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.faissId2MVId = []


    def add_content(self, content_embedding: np.array, content_id: str) -> None:  # (input: np.ndarray or str)
        """
        :param content_embedding: embedding having shape (N, EMB_SIZE)
        """
        assert len(content_embedding.shape) == 1, 'Expecting one content at a time'
        assert content_embedding.shape[-1] == EMB_SIZE, 'Expected embedding size of {}, got {}'.format(EMB_SIZE, content_embedding.shape[-1])
        content_embedding = content_embedding.reshape(1, -1)
        faiss.normalize_L2(content_embedding)
        self.index.add(content_embedding)
        self.faissId2MVId.append(content_id)


    def retrieve(self, query_embedding: np.array, k: int) -> Tuple[List[str], List[float]]:  # (input: np.ndarray or str)
        """
        :param query_embedding: np.ndarray having shape (EMB_SIZE,)
        :param k: retrieve top_k contents from the pool
        """
        query_embedding  = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        similarities, contents_idx = self.index.search(query_embedding, k)                 # 
        # assuming only one query
        similarities               = similarities[0]
        contents_idx               = contents_idx[0] #faiss internal indices
        #faiss results are already sorted from highest to lowest
        #sorted_idx               = np.argsort(similarity)[::-1]
        #mv_content_ids           = [self.faissId2MVId[idx] for idx in contents_idx[sorted_idx]]
        #sorted_similarities      = similarity[sorted_idx]
        mv_content_ids             = [self.faissId2MVId[idx] for idx in contents_idx]
        
        return mv_content_ids, similarities

app.config['Indexer']          = Indexer()
app.config['ClipEncoder']      = ClipEncoder()
app.config['ContentContainer'] = ContentContainer()

def load_image(input_image):
    if input_image.filename == '':
        print('Image must have a file name')
        return redirect(request.url)

    filename = secure_filename(input_image.filename)
    basedir = os.path.abspath(os.path.dirname(__file__))
    input_image.save(os.path.join(basedir, app.config["IMAGE_UPLOADS"], filename))

    img = Image.open(app.config["IMAGE_UPLOADS"]+"\\"+filename)
    data = io.BytesIO()
    img.save(data, 'JPEG')

    return base64.b64encode(data.getvalue())


@app.route("/")
def index():
	flash("Hello! This is Mediaverse retrieval system")
	return render_template("index.html")

@app.route('/mv_retrieval/v0.1/add_content', methods=['POST'])
def add_content():
    """
    Input is a json containing two fields

    :content_id: str
    :content_body:  Union[str, Image]

    """
    start_t                = datetime.now()

    # Empty content is equivalent to empty string
    input_text             = request.form['content_body_text']       # Input string
    input_image            = request.files['content_body_image']     # Input image

    #input_image = request.files.get('content_body_image', None)
    #print(input_image)
    #encode_img_data = load_image(input_image)
#
    #print(encode_img_data)

    if input_text != '' and input_image != '':
        flash('Please enter only one content at a time')
        return render_template("index.html")

    if input_text == '':
        input = input_image
        name = 'IMAGE'
    elif input_image == '':
        input = input_text
        name = 'TEXT'

    content_id             = str(request.form['content_id'])
    if input == '':
        flash('Please fill input box')
    elif content_id == '':
        flash('Please fill content id box')
    else:
        if not app.config['ContentContainer'].check_duplicate(input, content_id):
            content_embedding = app.config['ClipEncoder'].encode(input)
            app.config['Indexer'].add_content(content_embedding, content_id)
            end_t             = datetime.now()
            elapsed           = (end_t-start_t).total_seconds()
            flash('Content {} successfully added to the indexer in time {}. {}: {}'.format(content_id, elapsed, name, input))

    return render_template("index.html")#, filename = encode_img_data.decode('UTF-8'))


@app.route('/mv_retrieval/v0.1/retrieve_contents', methods=['POST'])
def retrieve():
    """
    Input is a json containing two fields

    :query:                  Union[str, Image]
    :n_contents_to_retrieve: int

    :return: return a payload with the fields 'contents' (List[str]) 
            and 'scores' (List[float])
    """
    input_text                  = request.form['query_text']       # Input text
    input_image                 = request.form['query_image']      # Input image

    if input_text != '' and input_image != '':
        flash('Please enter only one content at a time')
        return render_template("index.html")

    if input_text == '':
        input = input_image
    elif input_image == '':
        input = input_text

    k                      = request.form['n_contents_to_retrieve']
    if input == '':
        flash('[Retrieve] Please fill input query box')
    elif k == '':
        flash('[Retrieve] Please enter the number of contents to retrieve')
    else:
        k = int(k)
        query_embedding        = app.config['ClipEncoder'].encode(input)
        contents, similarities = app.config['Indexer'].retrieve(query_embedding, k)
        for content_id, score in zip(contents, similarities):
            content = app.config['ContentContainer'].get_text(content_id)
            flash('[Retrieve] Retrieved content {}: {}. Score: {}'.format(content_id, content, score))
    return render_template("index.html")


@app.route('/mv_retrieval/v0.1/all_contents', methods=['GET'])
def all_contents():
    app.config['ContentContainer'].getMVarchive()
    return render_template("index.html")