ARG BASE_REGISTRY=registry1.dso.mil
ARG BASE_IMAGE=ironbank/nextgen-federal/mistk/mistk-python
ARG BASE_TAG=1.2.0-3.11.8

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG}

WORKDIR /home/python
COPY . .

USER 1001
EXPOSE 30501

RUN pip install --no-cache-dir numpy==1.26.2 && \
# pip install --no-cache-dir sympy==1.13.0 && \ 
pip install --no-cache-dir pillow==10.3.0 && \
pip install --no-cache-dir typing_extensions==4.12.2 && \
pip install --no-cache-dir fastapi[standard] && \ 
pip install --no-cache-dir ultralytics==8.3.80 && \
pip install --no-cache-dir torch==2.2.0 && \
pip install --no-cache-dir torchvision==0.17 && \
pip install --no-cache-dir opencv-python-headless 
    
CMD ["fastapi", "run", "/home/python/Model/app.py", "--port", "30501"]

# add required HEALTH CHECK instruction ("NONE" since this is a base image)
HEALTHCHECK NONE
