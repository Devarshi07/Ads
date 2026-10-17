# Theraflo Ads service image: API, outbox relay, workers, and jobs share it.
#
#   docker build -t theraflo-ads .
#   docker compose --profile app up -d --build
#
# Multi-stage: wheels are built with the toolchain, the runtime stage has none.
# The runtime runs as an unprivileged user and holds no secrets; keys arrive as
# environment variables from the orchestrator's secret store.

FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[infra,ml,llm]"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    THERAFLO_ADS_LOG_FORMAT=json
# LightGBM needs the OpenMP runtime. /data/lake is created owned by the app user, so a
# fresh named volume mounted there (the analytics lake) inherits that ownership.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data/lake && chown app:app /data/lake
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels "theraflo-ads[infra,ml,llm]" && rm -rf /wheels
USER app
WORKDIR /home/app
EXPOSE 8000
CMD ["theraflo-ads", "serve", "--host", "0.0.0.0", "--port", "8000"]
