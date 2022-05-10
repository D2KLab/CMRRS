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
import os

import faiss
import numpy as np
from flask import Flask, jsonify, request
#from logstash_formatter import LogstashFormatterV1

import torch
import clip
from PIL import Image
from io import BytesIO
from collections import defaultdict

from operator import itemgetter
import random

app     = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_SIZE = 512
CONTAINER = './contents'

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

class Container:
    def __init__(self) -> None:
        """
        self.content is a dictionary of 
        ->
        key   : username
        value : dictionary  of  -> 
                                key    :   faiss_index
                                value  :   list(content_ids)
        """
        self.content = {}
    
    def from_idx_to_id(self, faiss_idx, username):
        assert isinstance(faiss_idx, list), "Expecting list of indexes"
        assert isinstance(username, list), "Expecting list of users"
        content_ids = [self.content[username][idx][-1] for idx in faiss_idx] # per ogni indice ritorniamo il content id più recente
        return content_ids

    def get_indexes(self, username):
        return list(self.content[username].keys())

    # def get_mvID(self, username):
    #     return self.content[username]["content_id"]
    
    # def get_userID(self):
    #     return list(self.content.keys())

    def add_content(self, faiss_idx, content_id, username):
        if username in list(self.content.keys()): 
            if faiss_idx in list(self.content[username].keys()):
                self.content[username][faiss_idx].append(content_id)
            else:
                self.content[username][faiss_idx] = [content_id]
        else:
            self.content[username] = {faiss_idx: [content_id]}

class ClipEncoder:
    def __init__(self) -> None:
        model, preprocess = clip.load("ViT-B/32") # (or load model starting from state_dict)
        self.model = model
        self.model.to(DEVICE).eval()
        self.preprocess = preprocess

    def encode(self, input: str, type: str) -> np.array:
        """
        input : binary text or image
        type  : str
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
        self.users        =  []
        self.idx_counter  =  0

    def get_embedding(self, indexes):
        if isinstance(indexes, list):
            return [self.index.reconstruct(idx) for idx in indexes] # controllare se idx è giusto così o ci va il -1 nel reconstruct
        elif isinstance(indexes, int):
            return self.index.reconstruct(indexes)
        

    def add_content(self, content_embedding: np.array, content_id: str, user_id: str, type: str) -> None:  # (input: np.ndarray or str)
        """
        :param content_embedding: embedding having shape (N, EMB_SIZE)
        """
        assert len(content_embedding.shape) == 1, 'Expecting one content at a time'
        assert content_embedding.shape[-1] == EMB_SIZE, 'Expected embedding size of {}, got {}'.format(EMB_SIZE, content_embedding.shape[-1])

        content_embedding = content_embedding.reshape(1, -1)
        faiss.normalize_L2(content_embedding)

        indexes = app.config['Container'].get_indexes(user_id)
        for idx in indexes:
            if self.index.reconstruct(idx) == content_embedding[0]:
                return idx

        self.index.add(content_embedding)
        self.idx_counter += 1
        self.users.append(user_id)
        return self.idx_counter
        

    def retrieve(self, query_embedding: np.array, k: int) -> Tuple[List[float], List[int]]:  # (input: np.ndarray or str)
        """
        :param query_embedding: np.ndarray having shape (EMB_SIZE,)
        :param k: retrieve top_k contents from the pool
        """
        # query_embedding  = query_embedding.reshape(1, -1).astype(np.float32)
        # query_embedding = self.index.reconstruct(idx)
        faiss.normalize_L2(query_embedding)
        similarities, contents_idx = self.index.search(query_embedding, k)
        # assuming only one query
        # similarities               = similarities[0]
        # contents_idx               = contents_idx[0] #faiss internal indices
        # mv_content_ids             = [self.faissId2MVId[idx] for idx in contents_idx]
        users_ids                  = [self.users[idx] for idx in contents_idx]
        return similarities[0], contents_idx[0], users_ids

app.config['Indexer'] = Indexer()
app.config['ClipEncoder'] = ClipEncoder()
app.config['Container'] = Container()

@app.route('/mv_retrieval/v0.1/add_content', methods=['POST'])
def add_content():
    """
    Input is a json containing two fields

    :content_id              : str
    :content                 : binary text or image
    :type (text or image)    : str

    """

    start_t           = datetime.now()

    content           = request.data
    content_id        = request.args['id']
    type              = request.args['type']
    user              = request.args['user']

    content_embedding = app.config['ClipEncoder'].encode(content, type)
    # rendo gli idx relativi a quell'user, faccio il reconstruct e controllo se gli embedding ci sono già
    

    # if tuple(content_embedding) in app.config['Container'].get_embeddings():
    #     elapsed           = (datetime.now()-start_t).total_seconds()
    #     out_msg           = {'msg': 'Content arleady present in the MV archive with id: {} and uploaded by user: {}'.format(app.config['Container'].get_mvID(tuple(content_embedding)), app.config['Container'].get_userID(tuple(content_embedding))),
    #                         'time': elapsed} 
    #     return jsonify(out_msg), 200

    faiss_idx = app.config['Indexer'].add_content(content_embedding, user)
    app.config['Container'].add_content(faiss_idx, content_id, user)

    
    end_t             = datetime.now()
    elapsed           = (end_t-start_t).total_seconds()
    out_msg           = {'msg': 'Content {} successfully added to the indexer by user {}'.format(content_id, user),
                         'time': elapsed} 
    return jsonify(out_msg), 200


@app.route('/mv_retrieval/v0.1/retrieve_contents', methods=['POST'])
def retrieve():
    """
    Input is a json containing two fields

    :content (query)                      : binary text or image
    :k (number of contents to retrieve)   : int
    :type (text or image)                 : str

    :return: return a payload with the fields 'contents' (List[str]) 
            and 'scores' (List[float])
    """
    
    k                 = int(request.args['k'])
    posting_user      = request.args['user']
    
    if (request.data):
        content           = request.data
        type              = request.args['type']
        query_embedding   = app.config['ClipEncoder'].encode(content, type)
        similarities, content_idx, users = app.config['Indexer'].retrieve(query_embedding, k)
        content_ids = app.config['Container'].from_idx_to_id(content_idx, users)
        return jsonify({'contents': content_ids, 'scores': similarities.tolist()})
    
    else:
        contents       = []
        similarities   = []
        # embeddings     = list(app.config['Container'].get_embeddings())
        # ids            = [app.config['Container'].get_mvID(embeddings[i]) for i in range(len(embeddings))]
        # users          = [app.config['Container'].get_userID(embeddings[i]) for i in range(len(embeddings))]

        #ALTERNATIVA: clusterizzare i contenuti con dbscan/hdbscan

        # vado a prendere gli indici faiss dei contenuti postati dall'utente
        # per ogni indice faccio un recostruct e prendo l'embedding
        idx_posted_contents = app.config['Container'].get_indexes(posting_user)
        embeddings = app.config['Indexer'].get_embeddings(idx_posted_contents)
        for embedding in embeddings:
            simil, indexes, user = app.config['Indexer'].retrieve(np.asarray(embedding), k)
            cont = app.config['Container'].from_idx_to_id(indexes, users)
            # keep only the recommended content that do not belong to the posting user
            contents.extend([c for (i,c) in enumerate(cont) if user[i] != posting_user])
            similarities.extend([s for (i,s) in enumerate(simil.tolist()) if user[i] != posting_user])
        
        # sort the general list of recommended contents by similarity index
        sort_indexes, similarities_sorted = zip(*sorted(enumerate(similarities), key=itemgetter(1)))
        contents = [contents[i] for i in sort_indexes]

        # manca distribuzione di probabilità (fare cumulative distribution function?)
        chosen_contents = random.choices(population = contents, weights = similarities, k = k)
        chosen_similarities = []
        for (i,element) in enumerate(chosen_contents): # trovare un metodo più efficace per fare questa ricerca
            i = contents.index(element)
            chosen_similarities.append(similarities[i])
        # comunque questo non va bene perchè ti può ritornare lo stesso contenuto più volte

        return jsonify({'contents': chosen_contents, 'scores': chosen_similarities}) # fare reverse





