"""

Flow:
    - Expose update_index and call it whenever there is a new content in the DB to add
    - Expose retrieve that returns sorted top_k given a query embedding

Doubts:
    - Faiss assigns an internal ID to each embedding in the `index` structure. We should build a mapping to the global ID of the content within the MV network
"""

import logging
from typing import List, Tuple
from datetime import datetime

import faiss
import numpy as np
from flask import Flask, jsonify, request
#from logstash_formatter import LogstashFormatterV1

import torch
import clip
from PIL import Image
from io import BytesIO
from collections import defaultdict

app     = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_SIZE = 512
CONTAINER = './contents'

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

class Container:
    def __init__(self) -> None:
        """
        content is a dictionary of
        key: embedding
        value: content id in mv
        """
        self.content = defaultdict(lambda: 'Not Present')

    def get_embeddings(self):
        return self.content.keys()

    def get_mvID(self, embedding):
        return self.content[embedding]

    def add_content(self, embedding, id):
        self.content[embedding] = id

class ClipEncoder:
    def __init__(self) -> None:
        model, preprocess = clip.load("ViT-B/32") # (or load model starting from state_dict)
        self.model = model
        self.model.to(DEVICE).eval()
        self.preprocess = preprocess

    def encode(self, input: str, type: str) -> np.array:
        """
        moment: 'query' - retrieval
        """
        if type == 'text':
            input = input.decode('UTF-8')
            text = clip.tokenize(input).to(DEVICE)
            with torch.no_grad():
                embedding = self.model.encode_text(text).detach().cpu().numpy().reshape(512).astype(np.float32)
        elif type == 'image':
            input = BytesIO(input)
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = self.model.encode_image(image).detach().cpu().numpy().reshape(512).astype(np.float32)
        else: 
            raise  ValueError(colored(255,0,0, 'Not valid type value, enter text or image'))

        return embedding
        
class Indexer:
    def __init__(self, emb_size: int=EMB_SIZE) -> None:
        # to get total length of flat index: index.xb.size()
        # to get number of embeddings in index: index.xb.size() // EMB_SIZE
        self.index        = faiss.index_factory(emb_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.faissId2MVId = []

    def add_content(self, content_embedding: np.array, content_id: str, type: str) -> None:  # (input: np.ndarray or str)
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
        similarities, contents_idx = self.index.search(query_embedding, k)
        # assuming only one query
        similarities               = similarities[0]
        contents_idx               = contents_idx[0] #faiss internal indices
        mv_content_ids             = [self.faissId2MVId[idx] for idx in contents_idx]
        return mv_content_ids, similarities

app.config['Indexer'] = Indexer()
app.config['ClipEncoder'] = ClipEncoder()
app.config['Container'] = Container()

@app.route('/mv_retrieval/v0.1/add_content', methods=['POST'])
def add_content():
    """
    Input is a json containing two fields

    :content_id: str
    :content_body:  str (text or base64 image)
    :type (text or image): str

    """

    start_t           = datetime.now()

    content           = request.data
    content_id        = request.args['id']
    type              = request.args['type']

    content_embedding = app.config['ClipEncoder'].encode(content, type)

    if tuple(content_embedding) in app.config['Container'].get_embeddings():
        elapsed           = (datetime.now()-start_t).total_seconds()
        out_msg           = {'msg': 'Content arleady present in the MV archive with id: {}'.format(app.config['Container'].get_mvID(tuple(content_embedding))),
                            'time': elapsed} 
        return jsonify(out_msg), 200
    
    app.config['Container'].add_content(tuple(content_embedding), content_id)

    app.config['Indexer'].add_content(content_embedding, content_id, type)
    end_t             = datetime.now()
    elapsed           = (end_t-start_t).total_seconds()
    out_msg           = {'msg': 'Content {} successfully added to the indexer'.format(content_id),
                         'time': elapsed} 
    return jsonify(out_msg), 200


@app.route('/mv_retrieval/v0.1/retrieve_contents', methods=['POST'])
def retrieve():
    """
    Input is a json containing two fields

    :query:                  str (text or base64 image)
    :n_contents_to_retrieve: int
    :type (text or image):   str

    :return: return a payload with the fields 'contents' (List[str]) 
            and 'scores' (List[float])
    """
    content           = request.data
    k                 = int(request.args['k'])
    type              = request.args['type']

    query_embedding        = app.config['ClipEncoder'].encode(content, type)
    contents, similarities = app.config['Indexer'].retrieve(query_embedding, k)
    return jsonify({'contents': contents, 'scores': similarities.tolist()})
