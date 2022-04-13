from flask import json
import numpy as np
import requests
import os
import random
import string
#import time

import skimage

update_url   = 'http://localhost:6000/mv_retrieval/v0.1/add_content'
retrieve_url = 'http://localhost:6000/mv_retrieval/v0.1/retrieve_contents'

descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer"
}

if __name__ == '__main__':
    texts  = list(descriptions.values())[1:]
    
    # LOAD TEXT
    # for content in texts:
    #     random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
    #     arg = '?id='+random_str+'&type=text'

    #     with open('tmpfile.txt', 'w') as f:
    #         f.write(content)
    #     with open('tmpfile.txt', 'rb') as f:
    #         data = f.read()
        
    #     response   = requests.post(update_url+arg, data=data)
    #     print('[UPDATE] -- status: {}, {}'.format(response.status_code, response.content))

    # LOAD IMAGES
    for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue
        
        with open(os.path.join(skimage.data_dir, filename), 'rb') as f:
            data = f.read()

        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
        if name == 'page':
            print(descriptions[name], '(image) :', random_str)
        arg = '?id='+random_str+'&type=image'
        response   = requests.post(update_url+arg, data=data)
        print('[UPDATE] -- status: {}, {}'.format(response.status_code, response.content))

    # QUERY
    query = "a page of text about segmentation"
    print('QUERY:', query)
    with open('tmpfile.txt', 'w') as f:
        f.write(query)
    with open('tmpfile.txt', 'rb') as f:
        data = f.read()

    arg = '?type=text&k=10'
    response = requests.post(retrieve_url+arg, data=data)
    print('[RETRIEVE]-- status: {}, {}'.format(response.status_code, response.content))