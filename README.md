[[CLIP]](https://github.com/openai/CLIP) [[FAISS]](https://github.com/facebookresearch/faiss)

This application exposes a REST service to make content retrieval (through a search query, text or image) or content recommendation based on hystorical contents precedently updated by the user.
The service makes use of a pretrained **CLIP** model with:
- Vit-B/32 as vision encoder (512 embeddings dimension, 224 input resolution, 12 layers, 12 heads)
- BERT as text encoder (63M-parameter 12- layer 512-wide model with 8 attention heads).

Contents are indexed and searched in the application node by means of **faiss** library which speed up similarity search over the representational space.

## Retrieval
![APP](retrievalsys.png)

The user can access the system by means of two functions:
- **add_content()**: in this way the user can add a content (text or image) to the system. It will be encoded to a 512 embedding vector and stored into the Faiss index. This function accepts one content at a time. 
    - Input:
        - username: unique identifier of the user who post the content;
        - text or image binary data (refer to usage.py example);
        - id: Mediaverse ID of the content to load;
        - type: "text" or "image", string describing the data type of the content to be loaded.
    - Output:
        - a success message in the field 'msg'
        - the elapsed time for the operation in the field 'time'

- **retrieve()**: the user enters a query (text or image) with the aim of retrieving as a result the k elements most similar, to the submitted 
query, among those stored inside the Faiss index. The top K contents are ranked according to the cosine similarity with respect to the input query. 
    - Input:
        - username: unique identifier of the user who search for a content (its contents are excluded from the retrieval process);
        - text or image binary data (refer to usage.py example);
        - k: number of similar contents to retrieve;
        - type: "text" or "imege", string describing the data type of the input query.
    - Output (json file with the follwoing hierarchy):
        - recommended texts in the field 'text':
            - 'contents' is an ordered list containing the ids (string) of the retrieved texts. The list is ordered based on decreasing values of similarity scores (i.e., the first content is the best one retrieved (among all texts) for that query)
            - 'scores' is an ordered list of similarity scores for the retrieved texts (i.e., the first score represents the similarity between the query and the first content in the 'contents' field).
        - recommended images in the field 'image':
            - 'contents' is an ordered list containing the ids (string) of the retrieved images. The list is ordered based on decreasing values of similarity scores (i.e., the first content is the best one retrieved (among all images) for that query)
            - 'scores' is an ordered list of similarity scores for the retrieved images (i.e., the first score represents the similarity between the query and the first content in the 'contents' field).
        
#### Retrieval output example
Given a query 'a page of text about segmentation' with k=4, the retrieve() fuction returns: 
```
[RETRIEVE]-- status: 200, b'{"image":{"contents":["4AYKRJ8QFS","HANEP78MN0","VJNL2OH70S","UVL3UWLG6Y"],"scores":[0.3587474226951599,0.22719718515872955,0.22674132883548737,0.22545325756072998]},"text":{"contents":["QICQ8T7NF9","TFGZJJ6UPX","FS5EI089C9","BGDTOVNTL5"],"scores":[0.6652600765228271,0.6554150581359863,0.619107723236084,0.6146003603935242]}}\n'
```

[[HDBSCAN]](https://github.com/scikit-learn-contrib/hdbscan)
## Recommendation
![APP](recsys.png)

The recommendation service is exposed by the recommend() function which resides on the same REST API of the retrieval system. Hence, the application has **only one container** exposing 3 functions: add_content(), retrieve, recommend().

The recommendation function makes use of user previous contents posted to create a seed which is then given as input to the search system. The recommended contents are required to satisfy: similarity with respect to the user seed and a certain degree of diversity compared to his post history (show to the user new contents). 

**HDBSCAN** was used to build the user seed. This technique uses density of neighbouring points to construct clusters, allowing clusters of any shape to be identified. User previous posts are clustered and new contents are recommended distinguishing from three cases: 0 clusters, 1 cluster, 2 or more clusters.

- **recommend()**: the user does not enter any query. Starting from a seed, the System suggests to the user new contents based on user post history and a certain degree of novelty. The top K contents are ranked according to the cosine similarity with respect to the generated seed.
    - Input:
        - username: unique identifier of the user who search for a content (its contents are excluded from the recommendation process);
        - k: number of similar contents to recommend; 
    - Output (It is divided into 2 sections):
        - 'text' (if the user seed is build starting from text contents):
            - (text2text) - recommended texts in the field 'text':
                - 'contents' is an ordered list containing the ids (string) of the retrieved texts. The list is ordered based on decreasing values of similarity scores (i.e., the first content is the best one retrieved (among all texts) for that query).
                - 'scores' is an ordered list of similarity scores for the retrieved texts (i.e., the first score represents the similarity between the query and the first content in the 'contents' field).
            - (text2image) - recommended images in the field 'image':
                - 'contents' is an ordered list containing the ids (string) of the retrieved images. The list is ordered based on decreasing values of similarity scores (i.e., the first content is the best one retrieved (among all images) for that query).
                - 'scores' is an ordered list of similarity scores for the retrieved images (i.e., the first score represents the similarity between the query and the first content in the 'contents' field).
        - 'image' (if the user seed is built from images):
            - (image2text) - recommended texts in the field 'text':
                - 'contents': same as above.
                - 'scores': same as above.
            - (image2image) - recommended images in the field 'image':
                - 'contents': same as above.
                - 'scores': same as above.

#### Recommendation output example
Given k=3, the recommend() fuction returns:
```
[RECOMMEND] -- status: 200, b'{"image":{"image":{"contents":["EY41X06ZQ0","COHG9FZSBC","9BG7DRB8PJ"],"scores":[0.6429423689842224,0.6368139386177063,0.5770949125289917]},"text":{"contents":["RCJ7ZGN4JX","NM5TCXJ4J4","ZC2Q2MY6PC"],"scores":[0.2598402500152588,0.22304044663906097,0.1921837329864502]}},"text":{"image":{"contents":["ZTL4B5A68V","BB4MVFIY92","9BG7DRB8PJ"],"scores":[0.17730148136615753,0.17051610350608826,0.1601783037185669]},"text":{"contents":["MXITHTY3Z5","P2IFRXWTBL","H4VARIWUND"],"scores":[0.6838483214378357,0.6227728128433228,0.6023468375205994]}}}
```

## Expose the service
#### 1. Manual setup

```
conda create --name mediaverse_rest python=3.8
conda activate mediaverse_rest
conda install -c pytorch faiss-cpu
pip install -r requirements.txt
flask run --port <PORT>
```

#### 2. Run a Docker container

```
cd docker
docker build -t flask-restapi .
docker tag flask-restapi mediaverse/flask-restapi
docker rmi flask-restapi

docker run -d --network host --name mediaverse_rest mediaverse/flask-restapi
```

As the instrutions show it is suggested to run the application in the host network providing an available port (settable in the Dockerfile).

#### 3. Docker-compose up
```
cd docker
docker-compose up --build
```