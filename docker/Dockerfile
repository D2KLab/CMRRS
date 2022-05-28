FROM continuumio/miniconda3
USER root
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
# Create the environment
RUN conda create --name mediaverse_rest python=3.8
RUN echo "source activate mediaverse_rest" > ~/.bashrc
ENV PATH /opt/conda/envs/mediaverse_rest/bin:$PATH
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "mediaverse_rest", "/bin/bash", "-c"]
RUN conda install -c pytorch faiss-cpu
RUN conda config --add channels conda-forge
RUN conda install -c conda-forge hdbscan
RUN pip install -r /app/requirements.txt

COPY ./app.py /app/app.py
CMD ["/app/scripts/start_api.sh"]