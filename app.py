"""

Flow:
    - Expose update_index and call it whenever there is a new content in the DB to add
    - Expose retrieve that returns sorted top_k given a query embedding

Doubts:
    - Faiss assigns an internal ID to each embedding in the `index` structure. We should build a mapping to the global ID of the content within the MV network
"""


import argparse
import logging
import os
from typing import List, Tuple
from datetime import datetime

import faiss
import numpy as np
from flask import Flask, jsonify, request
from logstash_formatter import LogstashFormatterV1

import torch
import clip
from PIL import Image

app     = Flask(__name__)
EMB_SIZE = 512
model, preprocess = clip.load("ViT-B/32") # (or load model starting from state_dict)

app.config['Clip'] = model
app.config['Preprocess'] = preprocess 

def encode(input):
    if type(input) is str:
        text = clip.tokenize(input)
        embs = app.config['Clip'].encode_text(text)
    else:
        image = app.config['Preprocess'](Image.open(input)).unsqueeze(0)
        embs = app.config['Clip'].encode_image(image)

    return embs

class Indexer:
    def __init__(self, emb_size: int=EMB_SIZE):
        # to get total length of flat index: index.xb.size()
        # to get number of embeddings in index: index.xb.size() // EMB_SIZE
        self.index        = faiss.index_factory(emb_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.faissId2MVId = []

    def add_content(self, input, content_id: str):  # (input: np.ndarray or str)
        """
        :param content_embedding: embedding having shape (N, EMB_SIZE)
        """
        content_embedding = encode(input)

        assert len(content_embedding.shape) == 1, 'Expecting one content at a time'
        assert content_embedding.shape[-1] == EMB_SIZE, 'Expected embedding size of {}, got {}'.format(EMB_SIZE, content_embedding.shape[-1])
        content_embedding = content_embedding.reshape(1, -1)
        faiss.normalize_L2(content_embedding)
        self.index.add(content_embedding)
        self.faissId2MVId.append(content_id)

    def retrieve(self, input, k: int) -> Tuple[List[str], List[float]]:  # (input: np.ndarray or str)
        """
        :param query_embedding: np.ndarray having shape (EMB_SIZE,)
        :param k: retrieve top_k contents from the pool
        """
        query_embedding = encode(input)

        query_embedding            = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        similarities, contents_idx = self.index.search(query_embedding, k)
        # assuming only one query
        similarities               = similarities[0]
        contents_idx               = contents_idx[0] #faiss internal indices
        #faiss results are already sorted from highest to lowest
        #sorted_idx               = np.argsort(similarity)[::-1]
        #mv_content_ids           = [self.faissId2MVId[idx] for idx in contents_idx[sorted_idx]]
        #sorted_similarities      = similarity[sorted_idx]
        mv_content_ids             = [self.faissId2MVId[idx] for idx in contents_idx]
        
        return mv_content_ids, similarities

app.config['Indexer'] = Indexer()

@app.route('/mv_retrieval/v0.1/add_content', methods=['POST'])
def add_content():
    """
    Input is a json containing two fields

    :content_id: str
    :embedding:  List[float]

    """
    start_t           = datetime.now()
    input             = request.get_json()
    content_embedding = np.array(input['embedding']).astype(np.float32)
    content_id        = input['content_id']
    app.config['Indexer'].add_content(content_embedding, content_id)
    end_t             = datetime.now()
    elapsed           = (end_t-start_t).total_seconds()
    out_msg           = {'msg': 'Content {} successfully added to the indexer'.format(content_id),
                         'time': elapsed} 
    return jsonify(out_msg), 200


@app.route('/mv_retrieval/v0.1/retrieve_contents', methods=['POST'])
def retrieve():
    """
    Input is a json containing two fields

    :query:                  List[float]
    :n_contents_to_retrieve: int

    :return: return a payload with the fields 'contents' (List[str]) 
            and 'scores' (List[float])
    """
    input                  = request.get_json()
    query_embedding        = np.array(input['query']).astype(np.float32)
    k                      = input['n_contents_to_retrieve']
    contents, similarities = app.config['Indexer'].retrieve(query_embedding, k)
    return jsonify({'contents': contents, 'scores': similarities.tolist()})
