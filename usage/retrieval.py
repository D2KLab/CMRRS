import numpy as np
import requests
import os
import random
import string

import skimage

env_vars = '../docker/.env'
for var in open(env_vars):
    key, value = var.split('=')
    os.environ[key] = value

PORT = os.environ.get('PORT') 

update_url   = f'http://localhost:{PORT}/mv_retrieval/v0.1/add_content'
retrieve_url = f'http://localhost:{PORT}/mv_retrieval/v0.1/retrieve_contents'

descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer", }
#     "coins": "a collection of coins",
#     "logo": "a green and orange logo with a snake",
#     "colorwheel": "colors distributed on a circle"
# }

if __name__ == '__main__':
    texts  = list(descriptions.values())[1:]
    
    print('-------------LOAD TEXT IN MEDIAVERSE-------------')
    for content in texts:
        username = 'mario.rossi'
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
        arg = '?id='+random_str+'&type=text&user='+username

        with open('tmpfile.txt', 'w') as f:
            f.write(content)
        with open('tmpfile.txt', 'rb') as f:
            data = f.read()
        
        response   = requests.post(update_url+arg, data=data)
        print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))

    print('-------------LOAD IMAGES IN MEDIAVERSE-------------')
    for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
        username = 'mario.rossi'
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
        arg = '?id='+random_str+'&type=image&user='+username

        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue
        with open(os.path.join(skimage.data_dir, filename), 'rb') as f:
            data = f.read()

        response   = requests.post(update_url+arg, data=data)
        print('[UPLOAD] -- status: {}, {}'.format(response.status_code, response.content))

    print('-------------QUERY-------------')
    username = 'john.smith'
    query = "a page of text about segmentation"
    print('QUERY:', query)
    with open('tmpfile.txt', 'w') as f:
        f.write(query)
    with open('tmpfile.txt', 'rb') as f:
        data = f.read()
    os.remove('tmpfile.txt')
    arg = '?type=text&k=3&user='+username
    response = requests.post(retrieve_url+arg, data=data)
    print('[RETRIEVE]-- status: {}, {}'.format(response.status_code, response.content))