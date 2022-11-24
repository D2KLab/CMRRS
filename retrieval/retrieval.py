import os
import random
import string
import csv

import sys
sys.path.insert(0,'.')

import numpy as np
import torch

# sys.path.append('/home/mediaverse/Adv-CLIP/retrieval')
from indexer import Indexer

from PIL import Image

PREFIX = '..'
EMB_SIZE = 768

sys.path.append('/home/mediaverse/Bridge-uni-multi-gap/datasets/flickr30k')
sys.path.append('/home/mediaverse/Bridge-uni-multi-gap/datasets/mscoco/dataextractors')
sys.path.append('/home/mediaverse/Bridge-uni-multi-gap/similarity_study')
from models.base_ViTBERT768 import BaseViTBERT
from mscoco_data import Captions
from flickr_data import FlickrCaptions

from transformers import ViTFeatureExtractor, ViTModel
from transformers import BertTokenizer, BertModel

def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def create_dirs(modelname):
#         for dataset in ['flickr', 'mscoco']:
#             for domain in ['similarity', 'retrieval']:
#                 if domain == 'similarity':
#                     for type in ['images', 'htmls', 'tables']:
#                         create_path_if_not_existant(os.path.join(PREFIX, 'results', dataset, modelname,  domain, type))
#                 else:
#                     create_path_if_not_existant(os.path.join(PREFIX, 'results', dataset, modelname, domain))

def retrieval_(dataset: torch.utils.data.Dataset,
               params, 
               sample_i: int,  
               task: str,  # txt2img, img2txt 
               model: torch.nn.Module,
               outputfile: str,
               n:int=1000, emb_size: int=EMB_SIZE, similarities=None):
    
    assert task in ['txt2img', 'img2txt'], 'Insert txt2img or img2txt as task'

    size          = dataset.size
    index         = Indexer(emb_size=emb_size)
    SAMPLES       = n
    N_RELEVANT    = 1

    # create_dirs(model_name)

    index2id      = {}  # sample index: img_id or txt_id
    
    # LOAD DATA
    for idx in random.sample(range(size), SAMPLES):
        txt, _, img, _ = dataset.__getitem__(idx)  # , _

        # Images loading
        if task == 'txt2img':
            img                        = img.cuda() 
            if model: img              = model.encode_image(img).squeeze()
            else:     img              = img.squeeze() 
            # Load image
            img_id               = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
            index.add_content(img.detach().cpu().numpy(), img_id)
            print('[UPDATE] -- {}'.format(img_id))
            index2id[idx]   = img_id

        # Texts loading
        if task == 'img2txt':
            txt                        = txt.cuda()
            if model: txt              = model.encode_text(txt).squeeze()
            else:     txt              = txt.squeeze() 
            # Load image
            txt_id               = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
            index.add_content(txt.detach().cpu().numpy(), txt_id)
            print('[UPDATE] -- {}'.format(txt_id))
            index2id[idx]        = txt_id
        
    # QUERIES
    for k in [10]:
        RECALL      = 0
        PRECISION   = 0
        MRR         = 0
        for idx, relevantid in index2id.items():
            txt, _, img, _ = dataset.__getitem__(idx)  # , _

            # text query
            if task == 'txt2img':
                txt                         = txt.cuda()  
                if model: txt               = model.encode_text(txt).squeeze()
                else:     txt               = txt.squeeze() 
                query                       = txt.detach().cpu().numpy()
            # image query
            else:
                img                         = img.cuda()
                if model: img               = model.encode_image(img).squeeze()
                else:     img               = img.squeeze()
                query                       = img.detach().cpu().numpy()

            retrieved_ids, scores = index.retrieve(query, k)

            relevant = 0
            mrr      = 0
            for position, id in enumerate(retrieved_ids):
                if id == relevantid:
                    relevant += 1
                    mrr  = 1/(position+1) 
            recall       = relevant/N_RELEVANT
            RECALL      += recall/SAMPLES
            precison     = relevant/k
            PRECISION   += precison/SAMPLES
            MRR         += mrr/SAMPLES
        
        
        
        row     = [sample_i] 
        row.extend(list(params.values()))
        metrics = [task, k, SAMPLES, round(RECALL, 2), np.nan, np.nan, np.nan, np.nan, round(PRECISION, 2), round(MRR, 2), np.nan]
        row.extend(metrics)

        if similarities:
            # similarities = [[1,2,3][][][]]
            for percentiles in similarities:
                row.extend(percentiles)

        with open(os.path.join(PREFIX, 'results', dataset.name, outputfile), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(row)


def retrieval_mixed(dataset: str,
               params, 
               sample_i: int,
               model: torch.nn.Module,
               outputfile: str, 
               n:int=1000, emb_size: int=EMB_SIZE, similarities=None):
    
    assert dataset in ['mscoco', 'flickr'], 'Expected one of mscoco or flickr'

    index         = Indexer(emb_size=emb_size)
    SAMPLES       = n
    N_RELEVANT    = 2

    if dataset == 'mscoco':
        captions_json       = '/media/storage/ai_hdd/datasets/MSCOCO/annotations/captions_val2014.json'
        images_path         = '/media/storage/ai_hdd/datasets/MSCOCO/images'
        rawdata             = Captions(number_of_samples=10000, captions_json=captions_json, images_path=images_path, karpathy=False)
    else:
    # FLICKR
        rawdata             = FlickrCaptions() # Here

    vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model             = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    bert_tokenizer        = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model            = BertModel.from_pretrained("bert-base-uncased")

    index2id      = {}  # sample index: img_id or txt_id

    # caption_idxs  = [random.sample(range(5), 2) for i in range(SAMPLES)]

    counter       = 0
    for idx in random.sample(range(rawdata.__len__()), 2*SAMPLES):
        obj                = rawdata.data[idx] 
        captions           = obj[1]
        if len(captions) < 5:
            continue
        
        caption_idxs       = random.sample(range(5), 2)
        caption            = captions[caption_idxs[0]]
        inputs             = bert_tokenizer(caption, return_tensors="pt")
        outputs            = bert_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        txt                = torch.mean(last_hidden_states, 1)
        txt                = txt.cuda()
        txt                = model.encode_text(txt).squeeze()

        img_id             = obj[0]
        if dataset == 'mscoco':
            img                = Image.open(os.path.join(rawdata.images_path, img_id+'.jpg'))
        else:
            img                = Image.open(os.path.join(rawdata.images_path, img_id))
        img                = img.convert('RGB')
        inputs             = vit_feature_extractor(img, return_tensors="pt")
        with torch.no_grad():
            outputs        = vit_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        img                = torch.mean(last_hidden_states, 1)
        img                = img.cuda()
        img                = model.encode_image(img).squeeze()

        txt_id               = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
        index.add_content(txt.detach().cpu().numpy(), txt_id)
        print('[UPDATE] -- {}'.format(txt_id))

        img_id          = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
        index.add_content(img.detach().cpu().numpy(), img_id)
        print('[UPDATE] -- {}'.format(img_id))

        # print(idx)
        # idx is not rawdata sample idx
        index2id[(idx, caption_idxs[1])]   = [txt_id]
        index2id[(idx, caption_idxs[1])].append(img_id)

        counter += 1
        if counter == SAMPLES:
            break
    
    print(len(index2id))

    from tqdm import tqdm

    # QUERIES
    for k in [10]:
        RECALL         = 0 # soft
        RECALL_HARD    = 0
        RECALL_TXT     = 0
        RECALL_IMG     = 0
        RECALL_IMG2TXT = 0
        PRECISION      = 0
        MRR            = 0
        TXTBEFOREIMG   = 0

        hard_counter = 0
        for (idx, caption_id), (relevantids) in tqdm(index2id.items()):
            # print(idx, caption_id, relevantids)

            # text query
            captions                    = rawdata.data[idx][1]
            caption                     = captions[caption_id]
            inputs                      = bert_tokenizer(caption, return_tensors="pt")
            outputs                     = bert_model(**inputs)
            last_hidden_states          = outputs.last_hidden_state
            txt                         = torch.mean(last_hidden_states, 1)
            txt                         = txt.cuda()
            txt                         = model.encode_text(txt).squeeze()
            query                       = txt.detach().cpu().numpy()
            
            retrieved_ids, scores = index.retrieve(query, k)

            relevant          = 0
            relevant_txt      = 0
            relevant_img      = 0
            mrr               = 0
            txtbeforeimg      = 0
            txtbeforeimg_flag = False
            for position, id in enumerate(retrieved_ids):
                if id in relevantids:
                    relevant += 1

                    if relevantids.index(id) == 0: 
                        relevant_txt      += 1

                    if relevantids.index(id) == 1:                          
                        relevant_img      += 1

                    if relevant == 1:
                        mrr   = 1/(position+1) 
                        if relevantids.index(id) == 0: # the first relevant is a text
                            txtbeforeimg_flag = True
            
            # both contents are required to be present (hard case)
            if relevant == 2:
                if txtbeforeimg_flag: # consider txtbeforeimg only when both are present in the top 10
                    txtbeforeimg += 1
                recall        = relevant/N_RELEVANT
                RECALL_HARD  += recall/SAMPLES
                hard_counter += 1
            
            recall        = relevant/N_RELEVANT
            RECALL       += recall/SAMPLES

            recall_txt        = relevant_txt/1
            RECALL_TXT       += recall_txt/SAMPLES
            recall_img        = relevant_img/1
            RECALL_IMG       += recall_img/SAMPLES

            precison      = relevant/k
            PRECISION    += precison/SAMPLES
            MRR          += mrr/SAMPLES
            TXTBEFOREIMG += txtbeforeimg

            # image query
            img_id                 = rawdata.data[idx][0]
            if dataset == 'mscoco':
                img                = Image.open(os.path.join(rawdata.images_path, img_id+'.jpg'))
            else:
                img                = Image.open(os.path.join(rawdata.images_path, img_id))
            img                = img.convert('RGB')
            inputs             = vit_feature_extractor(img, return_tensors="pt")
            with torch.no_grad():
                outputs        = vit_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            img                = torch.mean(last_hidden_states, 1)
            img                = img.cuda()
            img                = model.encode_image(img).squeeze()
            query              = img.detach().cpu().numpy()
            
            retrieved_ids, scores = index.retrieve(query, k+1)

            relevant_img2txt        = 0
            for id in retrieved_ids:
                if id in relevantids:
                    if relevantids.index(id) == 0: 
                        relevant_img2txt      += 1

            recall_img2txt        = relevant_img2txt/1
            RECALL_IMG2TXT       += recall_img2txt/SAMPLES
        
        try:
            TXTBEFOREIMG /= hard_counter
        except:
            TXTBEFOREIMG  = 0
        # TXTBEFOREIMG = 1 - abs(TXTBEFOREIMG-0.5)/0.5
        task    = 'mixed'
        row     = [sample_i] 
        row.extend(list(params.values()))
        metrics = [task, k, counter, round(RECALL, 2), round(RECALL_HARD, 2), round(RECALL_TXT, 2), round(RECALL_IMG, 2), round(RECALL_IMG2TXT, 2), round(PRECISION, 2), round(MRR, 2), round(TXTBEFOREIMG, 2)]
        row.extend(metrics)

        if similarities:
            # similarities = [[1,2,3][][][]]
            for percentiles in similarities:
                row.extend(percentiles)

        with open(os.path.join(PREFIX, 'results', dataset, outputfile), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

if __name__ == '__main__':
    pass
    # dataset = 'mscoco'
    # retrieval(dataset, 'txt2img', n=1000)