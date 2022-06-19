FROM pytorch/torchserve:latest-gpu

USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y lilypond git && rm -rf /var/lib/apt/lists/* && cd /tmp # buildkit

USER model-server
COPY brazen-score/models.py brazen-score/brazen_score.py brazen-score/utils.py brazen-score/parameters.py serve/config.properties serve/handler.py serve/requirements.txt models/brazen-score.pth symposium_properties.pickle /home/model-server/

RUN torch-model-archiver \
  --model-name brazen-score \
  --version 1.0 \
  --model-file /home/model-server/brazen_score.py \
  --extra-files /home/model-server/utils.py,/home/model-server/parameters.py,/home/model-server/models.py \
  --serialized-file /home/model-server/brazen-score.pth \
  --handler /home/model-server/handler.py \
  --requirements-file /home/model-server/requirements.txt \
  --export-path /home/model-server/model-store
RUN rm /home/model-server/brazen-score.pth

CMD ["torchserve", \
     "--start", \
     "--ncs", \
     "--ts-config /home/model-server/config.properties", \
     "--model-store /home/model-server/model-store"]
