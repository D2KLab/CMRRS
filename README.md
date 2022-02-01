## Setup

```
conda create --name mediaverse_rest python=3.8
conda activate mediaverse_rest
conda install -c pytorch faiss-cpu
pip install -r requirements.txt
```

## Docker image Setup and Container Run

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