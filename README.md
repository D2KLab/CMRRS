
This application exposes a REST service to make content retrieval (through a search query, text or image) or content recommendation based on hystorical contents precedently updated by the user.

## Retrieval
![APP](restapi.PNG)

The user can access the system by means of two actions:
- **Post content**: in this way the user can add a content (text or  image) to the system. It will be encoded to a 512 embedding vector and stored into the Faiss index.
- **Search query**: the user enters a query (text or image) with the aim of retrieving  as a result the k elements most similar, to the submitted query, among those stored inside the  Faiss index. The top K contents are ranked according to the cosine similarity with respect to the input query.


## Expose the service
#### 1. Manual setup

```
conda create --name mediaverse_rest python=3.8
conda activate mediaverse_rest
conda install -c pytorch faiss-cpu
pip install -r requirements.txt
flask run --port <PORT>
```

#### Run a Docker container

```
docker build -t flask-restapi .
docker tag flask-restapi mediaverse/flask-restapi
docker rmi flask-restapi

docker run -d --network host --name mediaverse_rest mediaverse/flask-restapi

docker ps
docker stop mediaverse_rest
docker rm mediaverse_rest
docker rmi mediaverse/flask-restapi
```

As the instrutions show it is suggested to run the application in the host network providing an available port (settable in the Dockerfile).