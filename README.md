## Setup

```
conda create --name mediaverse_rest python=3.8
conda activate mediaverse_rest
conda install -c pytorch faiss-cpu
pip install -r requirements.txt
```

## Docker image Setup

```
docker build -t flask-restapi .
docker tag flask-restapi mediaverse/flask-restapi
docker rmi flask-restapi
docker run -d -p 6000:6000 --name mediaverse_rest mediaverse/flask-restapi
```
