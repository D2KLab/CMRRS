import json
import random
from typing import Tuple

from torch.utils.data import Dataset

import os
from PIL import Image
import matplotlib.pyplot as plt

class CocoCaptions(Dataset):
    """COCO class that return matching (capt, objects) pairs 50% of times

    Input
    - number_of_samples  : number of sample taken from MSCOCO dataset
    - captions_json      : path to the json containing the images' captions
    - images_path        : path to the folder containg MSCOCO images
    - karpathy           : if the MSCOCO split is the 'karpathy split'

    self.data            : list of samples in the format [imageid, [caption1, caption2, ..., caption n]]
    """
    def __init__(self, number_of_samples: int, captions_json: str, images_path: str, karpathy = False):
        super(CocoCaptions, self).__init__()
        self.images_path = images_path
        self.karpathy = karpathy
        
        with open(captions_json, 'r') as fp:
            capt_data = json.load(fp)

        if karpathy:
            self.data = [[str(item['cocoid']), [caption['raw'] for caption in item['sentences']]] for item in capt_data['images'][:number_of_samples]]
        else:
            temp   = [[item['caption'], str(item['image_id'])] for item in capt_data['annotations']][:number_of_samples]
            values = set(map(lambda x:x[1], temp))
            self.data = [[x, [y[0] for y in temp if y[1]==x]] for x in values] 

    def get_data(self):
        return self.data

    def _get_sample_file(self, index: int) -> Tuple[str, bytes]:
        item  = self.data[index]
        n_captions = len(item[1])
        rand = random.randint(0, n_captions-1)
        capt  = item[1][rand]
        img_id = item[0]
        with open(os.path.join(self.images_path, img_id+'.jpg'), 'rb') as f:
            image = f.read()
        
        with open('tmpfile.txt', 'w') as f:
            f.write(capt)
        with open('tmpfile.txt', 'rb') as f:
            caption = f.read()
        # image = Image.open(os.path.join(self.images_path, img_id+'.jpg'))
        return caption, image, img_id

    def _get_sample(self, index: int) -> Tuple[str, str]:
        item  = self.data[index]
        n_captions = len(item[1])
        rand = random.randint(0, n_captions-1)
        capt  = item[1][rand]
        img_id = item[0]
        image = Image.open(os.path.join(self.images_path, img_id+'.jpg'))
        return capt, image

    def __len__(self):
        return len(self.data)
    
    def is_karpathy(self):
        return self.karpathy

    def display_couples(self, n: int) -> None:
        plt.figure(figsize=(22,22))
        col, row = find_couple(n)
        for i in range(n):
            capt, image = self._get_sample(i)
            plt.subplot(row, col, i+1)
            plt.imshow(image)
            plt.title(f"{capt}", color = 'white')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()


def find_couple(n):
    i = 1
    j = 1
    pairs = []
    for i in range(n+1):
        for j in range(n+1):
            if i*j == n:
                pairs.append((i, j))

    pairs = sorted(pairs, key=lambda t: abs(t[0]-t[1]), reverse=False)
    pair = pairs[0]
    if pair[0] > pair[1]:
        col = pair[0]
        row = pair[1]
    else:
        col = pair[1]
        row = pair[0]
    return row, col