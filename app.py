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

from operator import itemgetter
import random
import bisect
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances

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
        # le variabili passate sono nd arrays
        # assert isinstance(faiss_idx, list), "Expecting list of indexes"
        # assert isinstance(username, list), "Expecting list of users"
        content_ids = []
        for (i,idx) in enumerate(faiss_idx):
            content_ids.append(self.content[username[i]][idx][-1]) # per ogni indice ritorniamo il content id più recente
        return content_ids

    def get_indexes(self, username):
        if username in list(self.content.keys()):
            return list(self.content[username].keys())
        else:
            return []

    # def get_mvID(self, username):
    #     return self.content[username]["content_id"]
    
    # def get_userID(self):
    #     return list(self.content.keys())

    def add_content(self, faiss_idx, content_id, username):
        if username in list(self.content.keys()): 
            if faiss_idx in list(self.content[username].keys()):
                self.content[username][faiss_idx].append(content_id)
                # print(self.content)
            else:
                self.content[username][faiss_idx] = [content_id]
                # print(self.content)
        else:
            self.content[username] = {faiss_idx: [content_id]}
            # print(self.content)

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
        
    def get_len_index(self):
        return self.index.ntotal

    def add_content(self, content_embedding: np.array, user_id: str, type: str) -> None:  # (input: np.ndarray or str)
        """
        :param content_embedding: embedding having shape (N, EMB_SIZE)
        """
        assert len(content_embedding.shape) == 1, 'Expecting one content at a time'
        assert content_embedding.shape[-1] == EMB_SIZE, 'Expected embedding size of {}, got {}'.format(EMB_SIZE, content_embedding.shape[-1])

        content_embedding = content_embedding.reshape(1, -1)
        faiss.normalize_L2(content_embedding)

        # check if the same content has already been posted by the user. In that case we reuse the faiss index
        indexes = app.config['Container_'+type].get_indexes(user_id)
        for idx in indexes:
            if np.array_equal(self.index.reconstruct(idx), content_embedding[0]):
                return idx

        self.index.add(content_embedding)
        self.idx_counter += 1
        self.users.append(user_id)
        return self.idx_counter-1
        

    def retrieve(self, query_embedding: np.array, k: int) -> Tuple[List[float], List[int]]:  # (input: np.ndarray or str)
        """
        :param query_embedding: np.ndarray having shape (EMB_SIZE,)
        :param k: retrieve top_k contents from the pool
        """
        query_embedding            = query_embedding.reshape(1, -1).astype(np.float32)
        # query_embedding = self.index.reconstruct(idx)
        faiss.normalize_L2(query_embedding)
        similarities, contents_idx = self.index.search(query_embedding, k)

        # faiss internal indices
        contents_idx               = contents_idx[0]
        users_ids                  = [self.users[idx] for idx in contents_idx]
        return similarities[0], contents_idx, users_ids

class Clusterer:
    def __init__(self, embeddings):
        self.clusterer = hdbscan.HDBSCAN()
        self.embeddings = embeddings

    def fit(self):
        self.clusterer.fit(self.embeddings)

    def get_n_clusters(self):
        return self.clusterer.labels_.max()
    
    def get_clusters_count(self):
        return {i: list(self.clusterer.labels_).count(i) for i in range(self.clusterer.labels_.max())}
    
    def get_medoid(self, cluster):
        '''
        Input  : int
        Output : embedding of the medoid
        
        How to calculate medoid:
        1. compute pairwise distance matrix
        2. compute column or row sum
        3. argmin to find medoid index
        '''
        cluster_embeddings=[]
        cluster_indexes = []
        for (i, x) in enumerate(self.clusterer.labels_):
            if x == cluster:
                cluster_embeddings.append(self.embeddings[i])
                cluster_indexes.append(i)
        dist_matrix = pairwise_distances(cluster_embeddings)
        medoid_index = np.argmin(dist_matrix.sum(axis=0))
        return self.embeddings[cluster_indexes[medoid_index]]
    
    def get_outlier(self):
        # ADDRESS THE NO OULIERS CASE
        indexes = [i for (i,idx) in enumerate(self.clusterer.labels_) if idx == -1]
        assert len(indexes) != 0, "no outliers for serendipity"
        chosen_idx = random.choice(indexes)
        print("Index of the outlier chosen as seed: ", chosen_idx)
        return self.embeddings[chosen_idx]


app.config['Indexer_image'] = Indexer()
app.config['ClipEncoder_image'] = ClipEncoder()
app.config['Container_image'] = Container()

app.config['Indexer_text'] = Indexer()
app.config['ClipEncoder_text'] = ClipEncoder()
app.config['Container_text'] = Container()

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

    content_embedding = app.config['ClipEncoder_'+type].encode(content, type)
    # rendo gli idx relativi a quell'user, faccio il reconstruct e controllo se gli embedding ci sono già
    

    # if tuple(content_embedding) in app.config['Container'].get_embeddings():
    #     elapsed           = (datetime.now()-start_t).total_seconds()
    #     out_msg           = {'msg': 'Content arleady present in the MV archive with id: {} and uploaded by user: {}'.format(app.config['Container'].get_mvID(tuple(content_embedding)), app.config['Container'].get_userID(tuple(content_embedding))),
    #                         'time': elapsed} 
    #     return jsonify(out_msg), 200
    faiss_idx = app.config['Indexer_'+type].add_content(content_embedding, user, type)
    app.config['Container_'+type].add_content(faiss_idx, content_id, user)

    
    end_t             = datetime.now()
    elapsed           = (end_t-start_t).total_seconds()
    out_msg           = {'msg': 'Content {} successfully added to the indexer by user {}'.format(content_id, user),
                         'time': elapsed} 
    return jsonify(out_msg), 200


def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights, k):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    chosen_population = []
    chosen_weights = []
    for i in range(k):
        x = random.random()
        idx = bisect.bisect(cdf_vals, x)
        chosen_population.append(population[idx])
        chosen_weights.append(weights[idx])
    return chosen_population, chosen_weights

@app.route('/mv_retrieval/v0.1/retrieve_contents', methods=['POST'])
def retrieve():
    posting_user      = request.args['user'] 
    content           = request.data
    type_query        = request.args['type']
    k                 = int(request.args['k'])

    types = ["text", "image"]
    query_embedding   = app.config['ClipEncoder_'+type_query].encode(content, type_query)

    output = {}
    for type in types:

        # Number of contents posted by the query user
        n = len(app.config['Container_'+type].get_indexes(posting_user))

        assert k <= app.config['Indexer_'+type].get_len_index() - n, "requesting a number of contents greater than number of contents available"

        simil, indexes, users = app.config['Indexer_'+type].retrieve(query_embedding, k+n)
        cont_ids = app.config['Container_'+type].from_idx_to_id(indexes, users)

        # Filter out contents retrieved from query user Collection
        content_ids = [c for (i,c) in enumerate(cont_ids) if users[i] != posting_user]
        similarities = [s for (i,s) in enumerate(simil.tolist()) if users[i] != posting_user]

        output[type] = {'contents': content_ids[:k], 'scores': similarities[:k]}
    return jsonify(output)


@app.route('/mv_retrieval/v0.1/recommend_contents', methods=['POST'])
def recommend():
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

    types = ["text", "image"]
    output = {}
    
    content           = request.data
    type_query        = request.args['type']
    query_embedding   = app.config['ClipEncoder_'+type_query].encode(content, type_query)
    
    for type in types: 
        contents       = []
        similarities   = []

        # vado a prendere gli indici faiss dei contenuti postati dall'utente
        # per ogni indice faccio un recostruct e prendo l'embedding
        idx_posted_contents = app.config['Container_'+type].get_indexes(posting_user)
        embeddings = app.config['Indexer_'+type].get_embedding(idx_posted_contents)
        n = len(embeddings)
        assert k <= app.config['Indexer_'+type].get_len_index() - n, "requesting a number of contents greater than number of contents available"

        # clusterizzo i contenuti
        clusterer = Clusterer(embeddings)
        clusterer.fit()

        
        # se n_clusters == 0: random choice with similarity index as weights
        # se n_clusters == 1: un seed dal cluster e uno dagli outlier la maggior parte dei contenuti dal principale e tipo il 10% dagli outliers
        # se n_clusters >= 2: due seed dai due cluster principali e uno dagli outliers

    
        # if there is no clusters
        if not clusterer.get_clusters_count():
            print("zero clusters")
            # ---------------- Random choice with similarity index as weights ------------------------
            # NOT ONE SEED BUT SEVERAL SEEDS
            for embedding in embeddings:
                simil, indexes, users = app.config['Indexer_'+type].retrieve(np.asarray(embedding), k+n)
                cont = app.config['Container_'+type].from_idx_to_id(indexes, users)
                # keep only the recommended content that do not belong to the posting user
                contents.extend([c for (i,c) in enumerate(cont) if users[i] != posting_user])
                similarities.extend([s for (i,s) in enumerate(simil.tolist()) if users[i] != posting_user])
            
            sort_indexes, similarities_sorted = zip(*sorted(enumerate(similarities), key=itemgetter(1)))
            contents = [contents[i] for i in sort_indexes]

            chosen_contents, chosen_similarities = choice(contents, similarities_sorted, k = k)
            output[type] = {'contents': chosen_contents, 'scores': chosen_similarities}
            # comunque questo non va bene perchè ti può ritornare lo stesso contenuto più volte

        elif len(clusterer.get_clusters_count()) == 1:
            print("un cluster")
            cluster_seed = clusterer.get_medoid(0)
            outlier_seed = clusterer.get_outlier()
            cluster_k = k - k//5
            outlier_k = k - cluster_k

            simil, indexes, users = app.config['Indexer_'+type].retrieve(np.asarray(cluster_seed), cluster_k+n)
            cont = app.config['Container_'+type].from_idx_to_id(indexes, users)
            contents.extend([c for (i,c) in enumerate(cont) if users[i] != posting_user and i < k])
            similarities.extend([s for (i,s) in enumerate(simil.tolist()) if users[i] != posting_user and i < k])

            simil, indexes, users = app.config['Indexer_'+type].retrieve(np.asarray(outlier_seed), outlier_k+n)
            cont = app.config['Container_'+type].from_idx_to_id(indexes, users)
            contents.extend([c for (i,c) in enumerate(cont) if users[i] != posting_user and i < k])
            similarities.extend([s for (i,s) in enumerate(simil.tolist()) if users[i] != posting_user and i < k])

            sort_indexes, similarities_sorted = zip(*sorted(enumerate(similarities), key=itemgetter(1)))
            contents_sorted = [contents[i] for i in sort_indexes]

            output[type] = {'contents': contents_sorted, 'scores': similarities_sorted}


        elif len(clusterer.get_clusters_count()) >= 2:
            print("due o più")
            cluster_seed1 = clusterer.get_medoid(0)
            cluster_seed2 = clusterer.get_medoid(1)
            outlier_seed = clusterer.get_outlier()
            # outlier_k = int(k/10)
            # cluster2_k = (k - outlier_k) // 3
            # cluster1_k = k - cluster2_k - outlier_k
            outlier_k = k
            cluster2_k = k
            cluster1_k = k

            # bisogna fare che per gli embeddings delle imagini ritorniamo sia immagini che testo e per gli embeddings dei testi idem

            simil, indexes, users = app.config['Indexer_'+type].retrieve(np.asarray(cluster_seed1), cluster1_k+n)
            cont = app.config['Container_'+type].from_idx_to_id(indexes, users)
            contents.extend([c for (i,c) in enumerate(cont) if users[i] != posting_user])
            similarities.extend([s for (i,s) in enumerate(simil.tolist()) if users[i] != posting_user])

            simil, indexes, users = app.config['Indexer_'+type].retrieve(np.asarray(cluster_seed2), cluster2_k+n)
            cont = app.config['Container_'+type].from_idx_to_id(indexes, users)
            contents.extend([c for (i,c) in enumerate(cont) if users[i] != posting_user])
            similarities.extend([s for (i,s) in enumerate(simil.tolist()) if users[i] != posting_user])

            simil, indexes, users = app.config['Indexer_'+type].retrieve(np.asarray(outlier_seed), outlier_k+n)
            cont = app.config['Container_'+type].from_idx_to_id(indexes, users)
            contents.extend([c for (i,c) in enumerate(cont) if users[i] != posting_user])
            similarities.extend([s for (i,s) in enumerate(simil.tolist()) if users[i] != posting_user])

            sort_indexes, similarities_sorted = zip(*sorted(enumerate(similarities), key=itemgetter(1)))
            contents_sorted = [contents[i] for i in sort_indexes]

            output[type] = {'contents': contents_sorted[-k:], 'scores': similarities_sorted[-k:]} # fare reverse

        return jsonify(output) 






