FROM python:bookworm as build

WORKDIR /app
VOLUME [ "/data" ]
ARG TARGETPLATFORM
RUN echo "I'm building for $TARGETPLATFORM"
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt update && apt install -y ffmpeg

RUN . /opt/venv/bin/activate && pip install --no-cache-dir --upgrade pip

COPY ./requirements.txt /app

RUN . /opt/venv/bin/activate && pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY . /app

#RUN sh /app/test.sh

ENTRYPOINT  ["uvicorn","main:app","--host","0.0.0.0","--port","80","--reload","--reload-exclude *tmp"]
