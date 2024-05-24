FROM python:3.11-bullseye

WORKDIR /app

COPY . /app

RUN pip3 install -r /app/requirements.txt

ENTRYPOINT ["/usr/local/bin/python"]

CMD ["--help"]