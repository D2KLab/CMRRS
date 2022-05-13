
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
import bisect

import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
from data import CocoCaptions
from sklearn.metrics.pairwise import pairwise_distances


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_SIZE = 512

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

# captions_json  = '/media/storage/ai_hdd/datasets/MSCOCO/annotations/captions_val2014.json'

# images_path = '/media/storage/ai_hdd/datasets/MSCOCO/images'

# data  = CocoCaptions(number_of_samples=1000, captions_json=captions_json, images_path=images_path, karpathy=False)
# # print(data.get_data())
# SIZE = data.__len__()
# print('Number of samples: {}'.format(SIZE))

# encoder = ClipEncoder()
# embeddings_image = []
# embeddings_text = []

# for i in range(SIZE):
#     caption, image, img_id = data._get_sample_file(i)

#     emb = encoder.encode(image, "image")
#     emb = emb.reshape(1, -1)
#     faiss.normalize_L2(emb)
#     embeddings_image.append(emb[0])

#     emb = encoder.encode(caption, "text")
#     emb = emb.reshape(1, -1)
#     faiss.normalize_L2(emb)
#     embeddings_text.append(emb[0])

# # df = pd.DataFrame(embeddings_image)
# # print(df.shape)

# clusterer_text = hdbscan.HDBSCAN()
# clusterer_text.fit(embeddings_text)
# # print(clusterer_text.labels_)
# print("Numero di cluster captions: ",clusterer_text.labels_.max()+1)
# # print(clusterer_text.probabilities_)

# clusterer_image = hdbscan.HDBSCAN()
# clusterer_image.fit(embeddings_image)
# print("Numero di cluster images: ",clusterer_image.labels_.max()+1)
# # print(clusterer_image.labels_)
# # print(clusterer_image.probabilities_)

# classes = [i for i in range(clusterer_image.labels_.max()+1)]
# classes.append(-1)
# for classe in classes:
#     print("Captions per il cluster "+ str(classe))

#     # CALCOLA MEDOIDE
#     cluster_embeddings=[]
#     cluster_indexes = []
#     for (i, x) in enumerate(clusterer_image.labels_):
#         if x == classe:
#             cluster_embeddings.append(embeddings_image[i])
#             cluster_indexes.append(i)
#     dist_matrix = pairwise_distances(cluster_embeddings)
#     medoid_index = np.argmin(dist_matrix.sum(axis=0))
#     caption, image = data._get_sample(cluster_indexes[medoid_index])

#     plt.figure()
#     plt.imshow(image)
#     plt.savefig("./images/cluster"+str(classe)+"/medoid.png")
#     plt.show()

#     for i in range(SIZE):
#         if clusterer_image.labels_[i] == classe:
#             caption, image = data._get_sample(i)
#             plt.figure()
#             plt.imshow(image)
#             plt.savefig("./images/cluster"+str(classe)+"/image"+str(i)+".png")
#             plt.show()
#         if clusterer_text.labels_[i] == classe:
#             caption, image = data._get_sample(i)
#             print(caption)




# # DALL'OITPUT DEGLI ID RICOSTRUISCI LE IMMAGINI
captions_json  = '/media/storage/ai_hdd/datasets/MSCOCO/annotations/captions_train2014.json'

images_path = '/media/storage/ai_hdd/datasets/MSCOCO/images'

data  = CocoCaptions(number_of_samples=1000, captions_json=captions_json, images_path=images_path, karpathy=False)

ids =["480489","479495","12876","350553","187042","291827","465049","38682","309517","515040"]
indexes = []

images = data.get_data()
for id in ids:
    for (i,item) in enumerate(images):
        if item[0] == id:
            caption, image = data._get_sample(i)
            plt.figure()
            plt.imshow(image)
            plt.savefig("./images/output/"+id+".png")
            plt.show()















