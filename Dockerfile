FROM continuumio/miniconda3
WORKDIR /mediaverse
COPY requirements.txt .
# Create the environment
RUN conda create --name mediaverse_rest python=3.8
RUN echo "source activate mediaverse_rest" > ~/.bashrc
ENV PATH /opt/conda/envs/mediaverse_rest/bin:$PATH
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "mediaverse_rest", "/bin/bash", "-c"]
RUN conda install -c pytorch faiss-cpu
RUN pip install -r requirements.txt

COPY rest.py .
EXPOSE 6000
CMD ["python3", "rest.py", "--port", "6000"]