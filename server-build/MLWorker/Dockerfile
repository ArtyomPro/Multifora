FROM python:3.8

RUN mkdir -p /MLWorker

COPY . /MLWorker
WORKDIR /MLWorker

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT celery -A tasks worker -l info