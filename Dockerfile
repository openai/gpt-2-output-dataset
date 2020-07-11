FROM python:3.7.3-slim-stretch

WORKDIR /
COPY detector-base.pt /detector-base.pt
# TODO: keep root/.cache/torch/transformers/ between runs
# Make changes to the requirements/app here.
# This Dockerfile order allows Docker to cache the checkpoint layer
# and improve build times if making changes.
COPY requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt
COPY detector/ /detector

ENTRYPOINT ["python3", "-m", "detector.server", "detector-base.pt"]
