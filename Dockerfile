# Machina production image
# ARG BASE_REGISTRY=nexus-registry.dso-prod.machina.space
# ARG BASE_IMAGE=ironbank/nextgen-federal/mistk/mistk-python
# ARG BASE_TAG=1.2.0-3.11.8

#Temporary production image
ARG BASE_REGISTRY=registry1.dso.mil
ARG BASE_IMAGE=ironbank/nextgen-federal/mistk/mistk-python
ARG BASE_TAG=1.2.0-3.11.8

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG}

USER 0

RUN yum install mesa-libGL -y

ARG NEXUS_USER
ARG NEXUS_TOKEN
ARG PORT=30501
ARG WORKERS=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy necessary files for dependency installation
COPY uv.lock pyproject.toml /app/

# Install dependencies
RUN uv sync --no-cache --frozen --no-dev --no-install-project \
    --extra-index-url https://${NEXUS_USER}:${NEXUS_TOKEN}@nexus.dso-prod.machina.space/repository/pypi-all/simple

USER 1001

COPY ./Model /app/Model
COPY main.py /app
COPY app.py /app

EXPOSE ${PORT}

ENV PORT=${PORT}
ENV WORKERS=${WORKERS}

ENTRYPOINT uv run --no-dev uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers=${WORKERS}
