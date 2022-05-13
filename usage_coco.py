from flask import json
import numpy as np
import requests
import os
import random
import string
#import time

import skimage
from data import CocoCaptions

localhost = '8000'
update_url   = 'http://localhost:'+localhost+'/mv_retrieval/v0.1/add_content'
retrieve_url = 'http://localhost:'+localhost+'/mv_retrieval/v0.1/retrieve_contents'
recommend_url = 'http://localhost:'+localhost+'/mv_retrieval/v0.1/recommend_content'

# SIZE = 10

images_path = '/media/storage/ai_hdd/datasets/MSCOCO/images'

if __name__ == '__main__':
    
    # LOAD IMAGES ON MEDIAVERSE
    captions_json  = '/media/storage/ai_hdd/datasets/MSCOCO/annotations/captions_train2014.json'
    data  = CocoCaptions(number_of_samples=1000, captions_json=captions_json, images_path=images_path, karpathy=False)
    SIZE = data.__len__()
    # print("SIZE post di MV", SIZE) #241
    username = 'mario'
    for i in range(SIZE):
        caption, image, img_id = data._get_sample_file(i)
        arg = '?id='+img_id+'&type=image&user='+username
        response   = requests.post(update_url+arg, data=image)
        print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))


    # LOAD TEXT ON MEDIAVERSE
    captions_json  = '/media/storage/ai_hdd/datasets/MSCOCO/annotations/captions_train2014.json'
    data  = CocoCaptions(number_of_samples=1000, captions_json=captions_json, images_path=images_path, karpathy=False)
    SIZE = data.__len__()
    # print("SIZE post di MV", SIZE) #241
    username = 'mario'
    for i in range(SIZE):
        caption, image, img_id = data._get_sample_file(i)
        caption_id, image = data._get_sample(i)
        arg = '?id='+caption_id+'&type=text&user='+username
        response   = requests.post(update_url+arg, data=caption)
        print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))

    
    # LOAD USER PREVIOUS IMAGE POSTS
    captions_json  = '/media/storage/ai_hdd/datasets/MSCOCO/annotations/captions_val2014.json'
    data  = CocoCaptions(number_of_samples=1000, captions_json=captions_json, images_path=images_path, karpathy=False)
    SIZE = data.__len__()
    # print("SIZE post di luigi", SIZE)  #253
    username = 'luigi'
    for i in range(SIZE):
        caption, image, img_id = data._get_sample_file(i)
        arg = '?id='+img_id+'&type=image&user='+username
        response   = requests.post(update_url+arg, data=image)
        print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))

    # LOAD USER PREVIOUS TEXT POSTS
    captions_json  = '/media/storage/ai_hdd/datasets/MSCOCO/annotations/captions_val2014.json'
    data  = CocoCaptions(number_of_samples=1000, captions_json=captions_json, images_path=images_path, karpathy=False)
    SIZE = data.__len__()
    # print("SIZE post di MV", SIZE) #241
    username = 'luigi'
    for i in range(SIZE):
        caption, image, img_id = data._get_sample_file(i)
        caption_id, image = data._get_sample(i)
        arg = '?id='+caption_id+'&type=text&user='+username
        response   = requests.post(update_url+arg, data=caption)
        print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))

    
    # RECOMMENDATION
    username = 'luigi'
    arg = '?k=10&user='+username
    # response   = requests.post(recommend_url+arg, data=data)
    response   = requests.post(retrieve_url+arg)
    print('[RECOMMEND] -- status: {}, {}'.format(response.status_code, response.content))

    # QUERY
    # username = 'luigi'
    # query = "cats"
    # with open('tmpfile.txt', 'w') as f:
    #     f.write(query)
    # with open('tmpfile.txt', 'rb') as f:
    #     data = f.read()
    # arg = '?type=text&k=10&user='+username
    # response = requests.post(retrieve_url+arg, data=data)
    # print('[RETRIEVE]-- status: {}, {}'.format(response.status_code, response.content))
    
    # plt.figure()
    # plt.imshow(image)
    # plt.savefig("./images/medoid.png")
    # plt.show()






