from flask import json
import numpy as np
import requests
import pdb
#import time

update_url   = 'http://localhost:6000/mv_retrieval/v0.1/add_content'
retrieve_url = 'http://localhost:6000/mv_retrieval/v0.1/retrieve_contents'

if __name__ == '__main__':
    pool  = ["Fabio goes to the market", "Tommaso on the roof", "A dog on the street"]
    query = "Fabio back from the market"
    import random
    import string
    
    for content in pool:
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))    
        payload    = {'content_id': random_str, 'content_body': content}
        response   = requests.post(update_url,json=payload)
        print('[UPDATE] -- status: {}, {}'.format(response.status_code, response.content))
    payload  = {'query': query, 'n_contents_to_retrieve': 1}
    response = requests.post(retrieve_url, json=payload)
    print('[RETRIEVE]-- status: {}, {}'.format(response.status_code, response.content))