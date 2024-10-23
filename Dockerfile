FROM python:3.12

ARG PIP_NO_CACHE_DIR=True

WORKDIR /app

RUN apt-get update && apt-get install -y g++
RUN python -m pip install --upgrade pip setuptools wheel

COPY ./ .

WORKDIR /app/python

RUN python -m pip install -r requirements.txt
RUN python -m pip install .

WORKDIR /app

CMD python -c "import lorannlib; print('LoRANN has been installed')"
