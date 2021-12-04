FROM python:3.8

RUN mkdir -p /FlaskApi

COPY . /FlaskApi
WORKDIR /FlaskApi

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["service.py"]