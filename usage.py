from flask import json
import numpy as np
import requests
import os
import random
import string
#import time

import skimage



localhost = '8000'
update_url   = 'http://localhost:'+localhost+'/mv_retrieval/v0.1/add_content'
retrieve_url = 'http://localhost:'+localhost+'/mv_retrieval/v0.1/retrieve_contents'
recommend_url = 'http://localhost:'+localhost+'/mv_retrieval/v0.1/recommend_content'

# images previously uploaded by the user
previous_posts = {
    "horse": "a black-and-white silhouette of a horse",
    "page": "a page of text about segmentation"
    # "coins": "a collection of coins",
    # "logo": "a green and orange logo with a snake",
    # "colorwheel": "colors distributed on a circle"
}

# altri contenuti non caricati dall'utente
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad"
    # "motorcycle_right": "a red motorcycle standing in a garage",
    # "camera": "a person looking at a camera on a tripod",
    # "horse": "a black-and-white silhouette of a horse", 
    # "coffee": "a cup of coffee on a saucer",
    # "coins": "a collection of coins",
    # "logo": "a green and orange logo with a snake",
    # "colorwheel": "colors distributed on a circle"
}

if __name__ == '__main__':
    texts  = list(descriptions.values())
    previous_texts = list(previous_posts.values())
    
    # LOAD TEXT IN MEDIAVERSE
    # for content in texts:
    #     username = 'mario'
    #     #random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
    #     arg = '?id='+content+'&type=text&user='+username

    #     with open('tmpfile.txt', 'w') as f:
    #         f.write(content)
    #     with open('tmpfile.txt', 'rb') as f:
    #         data = f.read()
        
    #     response   = requests.post(update_url+arg, data=data)
    #     print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))

    # # LOAD IMAGES
    # for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    #     name = os.path.splitext(filename)[0]
    #     if name not in descriptions:
    #         continue
        
    #     with open(os.path.join(skimage.data_dir, filename), 'rb') as f:
    #         data = f.read()

    #     random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
    #     if name == 'page':
    #         print(descriptions[name], '(image) :', random_str)
    #     arg = '?id='+random_str+'&type=image'
    #     response   = requests.post(update_url+arg, data=data)
    #     print('[UPDATE] -- status: {}, {}'.format(response.status_code, response.content))

    # LOAD IMAGES ON MEDIAVERSE
    for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue
        
        with open(os.path.join(skimage.data_dir, filename), 'rb') as f:
            data = f.read()

        username = 'mario'
        arg = '?id='+name+'&type=image&user='+username
        response   = requests.post(update_url+arg, data=data)
        print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))

    # # LOAD USER'S PREVIOUS IMAGES 
    # for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    #     name = os.path.splitext(filename)[0]
    #     if name not in previous_posts:
    #         continue     
        
    #     with open(os.path.join(skimage.data_dir, filename), 'rb') as f:
    #         data = f.read()
        
    #     username = 'luigi'
    #     arg = '?id='+name+'&type=image&user='+username
    #     response   = requests.post(update_url+arg, data=data)
    #     print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))

    # LOAD USER'S PREVIOUS TEXTS
    for content in texts:
        username = 'luigi'
        #random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
        arg = '?id='+content+'&type=text&user='+username

        with open('tmpfile.txt', 'w') as f:
            f.write(content)
        with open('tmpfile.txt', 'rb') as f:
            data = f.read()
        
        response   = requests.post(update_url+arg, data=data)
        print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))


    # POST REQUEST TO RECOMMENDATION SYSTEM
    username = 'luigi'
    arg = '?k=8&user='+username
    # response   = requests.post(recommend_url+arg, data=data)
    response   = requests.post(retrieve_url+arg)
    print('[RECOMMEND] -- status: {}, {}'.format(response.status_code, response.content))

    # # QUERY
    # query = "a page of text about segmentation"
    # print('QUERY:', query)
    # with open('tmpfile.txt', 'w') as f:
    #     f.write(query)
    # with open('tmpfile.txt', 'rb') as f:
    #     data = f.read()
    # arg = '?type=text&k=10'
    # response = requests.post(retrieve_url+arg, data=data)
    # print('[RETRIEVE]-- status: {}, {}'.format(response.status_code, response.content))


   