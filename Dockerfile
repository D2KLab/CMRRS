FROM continuumio/miniconda3
WORKDIR /mediaverse

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY requirements.txt .
# Create the environment
RUN conda create --name mediaverse_rest python=3.8
RUN echo "source activate mediaverse_rest" > ~/.bashrc
ENV PATH /opt/conda/envs/mediaverse_rest/bin:$PATH
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "mediaverse_rest", "/bin/bash", "-c"]
RUN conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0   ### for CLIP
RUN conda install -c pytorch faiss-cpu
RUN pip install -r requirements.txt

EXPOSE 9000
COPY app.py .
CMD ["flask", "run", "--port", "9000"]