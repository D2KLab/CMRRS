from flask import json
import numpy as np
import requests
import pdb

update_url   = 'http://localhost:6000/mv_retrieval/v0.1/add_content'
retrieve_url = 'http://localhost:6000/mv_retrieval/v0.1/retrieve_contents'

if __name__ == '__main__':
    pool  = np.random.rand(10, 512).astype(np.float32)
    query = np.random.rand(512).astype(np.float32)
    import random
    import string
    
    for content in pool:
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))    
        payload    = {'content_id': random_str, 'embedding': content.tolist()}
        response   = requests.post(update_url, json=payload)
        print('[UPDATE] -- status: {}, {}'.format(response.status_code, response.content))
    payload  = {'query': query.tolist(), 'n_contents_to_retrieve': 4}
    response = requests.post(retrieve_url, json=payload)
    print('[RETRIEVE]-- status: {}, {}'.format(response.status_code, response.content))