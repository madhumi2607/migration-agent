# Generic image for running a target repo's test suite.
# The repo is NOT baked in: it's mounted read-only at /src for each run, so one
# image serves every benchmark task.
FROM python:3.11-slim-bookworm

# git: some projects compute their version from git metadata at install time
# (setuptools_scm, hatch-vcs), so `pip install .` fails without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

# Run as an unprivileged user, with packages going into a venv that user owns.
RUN useradd --create-home sandbox \
    && python -m venv /venv \
    && mkdir /work /out \
    && chown -R sandbox /venv /work /out
ENV PATH="/venv/bin:$PATH"

COPY entrypoint.sh /usr/local/bin/run-tests
RUN chmod +x /usr/local/bin/run-tests

USER sandbox
WORKDIR /work
ENTRYPOINT ["run-tests"]
