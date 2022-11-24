"""

Flow:
    - Expose update_index and call it whenever there is a new content in the DB to add
    - Expose retrieve that returns sorted top_k given a query embedding

Doubts:
    - Faiss assigns an internal ID to each embedding in the `index` structure. We should build a mapping to the global ID of the content within the MV network
"""

from typing import List, Tuple
from datetime import datetime
import os

import seaborn as sns
import faiss
import numpy as np
from flask import Flask, jsonify, request

import sys
import logging
from logstash_formatter import LogstashFormatterV1

import torch
import clip
from PIL import Image
from io import BytesIO
from torch.nn import Softmax

from collections import defaultdict

from operator import itemgetter
import random
import bisect
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

app     = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_SIZE = 512
CONTAINER = './contents'

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

# configure logging
logging.basicConfig(filename='log/flask.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        # assert isinstance(faiss_idx, list), "Expecting list of indexes"
        # assert isinstance(username, list), "Expecting list of users"
        content_ids = []
        for (i,idx) in enumerate(faiss_idx):
            content_ids.append(self.content[username[i]][idx][-1]) # return only the latest content id (mediaverse id) 
        return content_ids

    def get_indexes(self, username):
        if username in list(self.content.keys()):
            return list(self.content[username].keys()) # return all faiss indexes associated to input username
        else:
            return []

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
            raise  ValueError(colored(255,0,0, "Not valid type value, enter 'text' or 'image'"))

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
    
    def get_id_from_embedding(self, content_embedding: np.array, user_id: str, type: str):
        indexes = app.config['Container_'+type].get_indexes(user_id)
        for idx in indexes:
            if np.array_equal(self.index.reconstruct(idx), content_embedding):
                id = app.config['Container_'+type].from_idx_to_id([idx], [user_id])
                return id

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
        self.clusterer = hdbscan.HDBSCAN(alpha=1.0)
        self.embeddings = embeddings

    def fit(self):
        self.clusterer.fit(self.embeddings)

    def get_labels(self):
        return self.clusterer.labels_
    
    def get_probabilities(self):
        return self.clusterer.probabilities_

    def get_n_clusters(self):
        return self.clusterer.labels_.max()
    
    def get_clusters_count(self):
        # print(self.clusterer.labels_.max())
        return {i: list(self.clusterer.labels_).count(i) for i in range(self.clusterer.labels_.max()+1)}
    
    def get_main_clusters(self):
        count = {i: list(self.clusterer.labels_).count(i) for i in range(self.clusterer.labels_.max()+1)}
        #duplico il count
        d = count
        # get key with max value
        max_key1 = max(d, key=d.get)
        del d[max_key1]
        max_key2 = max(d, key=d.get)
        # print([max_key1, max_key2])
        return [max_key1, max_key2]

    
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
        # print("Index of the outlier chosen as seed: ", chosen_idx)
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

    faiss_idx = app.config['Indexer_'+type].add_content(content_embedding, user, type)
    app.config['Container_'+type].add_content(faiss_idx, content_id, user)

    end_t             = datetime.now()
    elapsed           = (end_t-start_t).total_seconds()
    out_msg           = {'msg': 'Content {} successfully added to the indexer by user {}'.format(content_id, user),
                         'time': elapsed}
    return jsonify(out_msg), 200


def choice(population, weights, k):
    assert len(population) == len(weights)
    cdf_vals = np.exp(weights)/sum(np.exp(weights))
    chosen_population = []
    chosen_weights = []
    x_list = []
    i = 0
    while i < k:
        x = int(np.random.choice(len(population), 1, p=cdf_vals))
        if x not in x_list:
            chosen_population.append(population[x])
            chosen_weights.append(weights[x])
            x_list.append(x)
            i += 1
    return chosen_population, chosen_weights

def recommend_content(seed, k, posting_user, output_type):
    contents       = []
    similarities   = []
    simil, indexes, users = app.config['Indexer_'+output_type].retrieve(np.asarray(seed), k)
    cont = app.config['Container_'+output_type].from_idx_to_id(indexes, users)
    # keep only the recommended content that do not belong to the posting user and that are not already present in the list of recommended content
    for (i,c) in enumerate(cont):
        if users[i] != posting_user and c not in contents:
            contents.append(c)
            similarities.append(float(simil[i]))
    return contents, similarities

@app.route('/mv_retrieval/v0.1/retrieve_contents', methods=['POST'])
def retrieve():
    posting_user      = request.args['user'] 
    content           = request.data
    type_query        = request.args['type']
    k                 = int(request.args['k'])

    types             = ["text", "image"]
    query_embedding   = app.config['ClipEncoder_'+type_query].encode(content, type_query)

    output = {}
    for type in types:

        # Number of contents posted by the query user
        n     = len(app.config['Container_'+type].get_indexes(posting_user))
        total = app.config['Indexer_'+type].get_len_index()

        # No contents of type 'type' other than user's ones
        if total - n == 0:
            output[type] = 'No contents available other than {} ones'.format(posting_user)
            continue

        assert k <= total - n, f"Requesting a number of '{type}' contents greater than number of '{type}' contents available. Choose k <= {total-n}"

        simil, indexes, users = app.config['Indexer_'+type].retrieve(query_embedding, k+n)
        cont_ids              = app.config['Container_'+type].from_idx_to_id(indexes, users)

        # Filter out contents retrieved from query user Collection
        content_ids           = [c for (i,c) in enumerate(cont_ids) if users[i] != posting_user]
        similarities          = [s for (i,s) in enumerate(simil.tolist()) if users[i] != posting_user]

        output[type]          = {'contents': content_ids[:k], 'scores': similarities[:k]}
    return jsonify(output)


@app.route('/mv_retrieval/v0.1/recommend_contents', methods=['POST'])
def recommend():
    """
    Input is a json containing 

    :username                          : str
    :k (number of recommended items)   : int

    :return: return a payload with the fields 'contents' (List[str]) 
            and 'scores' (List[float]) for each retrieval mode
    """
    k                 = int(request.args['k'])
    posting_user      = request.args['user'] 
    d                 = 0.2 # percentage of k returned from outlier seed

    input_types = ["text", "image"]
    output_types = ["text", "image"]
    output = {"text": {}, "image": {}}

    
    for in_type in input_types: 
        # take the faiss indexes of the user's previous posts
        idx_posted_contents = app.config['Container_'+in_type].get_indexes(posting_user)
        # for each index reconstruct the embedding
        embeddings = app.config['Indexer_'+in_type].get_embedding(idx_posted_contents)
        n = len(embeddings)
        total = app.config['Indexer_'+in_type].get_len_index()

        # No contents of type 'type' other than user's ones
        # if total - n == 0:
        #     output[type] = 'No contents available other than {} ones'.format(posting_user)
        #     continue

        assert k <= total - n, f"Requesting a number of '{in_type}' contents greater than number of '{in_type}' contents available. Choose k <= {total-n}"

        # HDBSCAN clustering of the previous content
        clusterer = Clusterer(embeddings)
        clusterer.fit()

        # UNCOMMENT FOR A 2D REPRESENTATION OF THE CLUSTERING
        # projection = TSNE().fit_transform(embeddings)
        # color_palette = sns.color_palette()
        # cluster_colors = [color_palette[x] if x >= 0
        #                 else (0.5, 0.5, 0.5)
        #                 for x in clusterer.get_labels()]
        # cluster_member_colors = [sns.desaturate(x, p) for x, p in
        #                         zip(cluster_colors, clusterer.get_probabilities())]
        # plt.figure()
        # plt.title("HDBSCAN Clustering of the user's previous content "+type)
        # plt.axis("off")
        # plt.scatter(*projection.T, linewidth=0, c=cluster_member_colors, alpha=0.25)
        # plt.savefig("./images_val/input/clusterization_"+type+".png")

        
        # if n_clusters == 0: random choice with similarity index as weights
        # if n_clusters == 1: one seed from cluster and one from random outlier
        # if n_clusters >= 2: two seeds from the two main clusters (one each) and one from the outliers

    
        # if there are no cluster: random choice with similarity indexs as weights --> number of seeds = n
        if not clusterer.get_clusters_count():
            print("zero clusters "+ str(in_type))
            for out_type in output_types:

                for embedding in embeddings:
                    contents, similarities = recommend_content(embedding, k+n, posting_user, out_type)
                # select random content with a probability built on similarity scores
                chosen_contents, chosen_similarities = choice(contents, similarities, k = k)
                # sort the content according to similarity score
                sort_indexes, similarities = zip(*sorted(enumerate(chosen_similarities), key=itemgetter(1)))
                contents = [chosen_contents[i] for i in sort_indexes]

                output[in_type][out_type] = {'contents':  list(contents[::-1]), 'scores':  list(similarities[::-1])}

        elif len(clusterer.get_clusters_count()) == 1:
            print("un cluster " + str(in_type))

            # select one seed from the cluster and one from a random outlier
            cluster_seed = clusterer.get_medoid(0)
            outlier_seed = clusterer.get_outlier()
            # return a percentage of k from the outlier determined by parameter d
            outlier_k = int(k*d)
            cluster_k = k - outlier_k

            # UNCOMMENT to print the ids of the chosen seeds
            print("Chosen seed for the outliers: ", app.config['Indexer_'+in_type].get_id_from_embedding(outlier_seed, posting_user, in_type))
            print("Chosen seed for the cluster: ", app.config['Indexer_'+in_type].get_id_from_embedding(cluster_seed, posting_user, in_type))            
            
            for out_type in output_types:
                # retrieve content from cluster seed
                contents_cluster, similarities_cluster = recommend_content(cluster_seed, cluster_k+n, posting_user, out_type)
                contents = contents_cluster[:cluster_k]
                similarities = similarities_cluster[:cluster_k]
                # retrieve content from outlier seed
                contents_outlier, similarities_outlier = recommend_content(outlier_seed, outlier_k+n, posting_user, out_type)
                contents.extend(contents_outlier[:outlier_k])
                similarities.extend(similarities_outlier[:outlier_k])

                sort_indexes, similarities_sorted = zip(*sorted(enumerate(similarities), key=itemgetter(1)))
                contents_sorted = [contents[i] for i in sort_indexes]

                output[in_type][out_type] = {'contents':  list(contents_sorted[::-1]), 'scores':  list(similarities_sorted[::-1])}


        elif len(clusterer.get_clusters_count()) >= 2:
            print("due o più clusters " + in_type)
            contents       = []
            similarities   = []
            cluster_seed1 = clusterer.get_medoid(clusterer.get_main_clusters()[0])
            cluster_seed2 = clusterer.get_medoid(clusterer.get_main_clusters()[1])
            outlier_seed = clusterer.get_outlier()

            outlier_k = int(k*d)
            cluster2_k = (k - outlier_k)//2
            cluster1_k = k - cluster2_k - outlier_k

            # UNCOMMENT to print the ids of the chosen seeds
            print("Chosen seed for the outliers: ", app.config['Indexer_'+str(in_type)].get_id_from_embedding(outlier_seed, posting_user, in_type))
            print("Chosen seed for the cluster 1: ", app.config['Indexer_'+str(in_type)].get_id_from_embedding(cluster_seed1, posting_user, in_type))
            print("Chosen seed for the cluster 2: ", app.config['Indexer_'+str(in_type)].get_id_from_embedding(cluster_seed2, posting_user, in_type))

            for out_type in output_types:
                # retrieve content from main cluster seed
                contents_cluster1, similarities_cluster1 = recommend_content(cluster_seed1, cluster1_k+n, posting_user, out_type)
                contents = contents_cluster1[:cluster1_k]
                similarities = similarities_cluster1[:cluster1_k]
                # retrieve content from the second cluster seed
                contents_cluster2, similarities_cluster2 = recommend_content(cluster_seed2, cluster2_k+n, posting_user, out_type)
                contents.extend(contents_cluster2[:cluster2_k])
                similarities.extend(similarities_cluster2[:cluster2_k])
                # retrieve content from outlier seed
                contents_outlier, similarities_outlier = recommend_content(outlier_seed, outlier_k+n, posting_user, out_type)
                contents.extend(contents_outlier[:outlier_k])
                similarities.extend(similarities_outlier[:outlier_k])

                sort_indexes, similarities_sorted = zip(*sorted(enumerate(similarities), key=itemgetter(1)))
                contents_sorted = [contents[i] for i in sort_indexes]

                output[in_type][out_type] = {'contents':  list(contents_sorted[::-1]), 'scores':  list(similarities_sorted[::-1])}

    return jsonify(dict(output))






