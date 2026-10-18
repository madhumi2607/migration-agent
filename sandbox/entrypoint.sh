#!/bin/sh
# Runs inside the sandbox container. Inputs (env vars, set by the sandbox server):
#   INSTALL_CMD  installs the repo and its test deps, e.g. "pip install . pytest pytest-mock"
#   UPGRADE      pip requirement(s) forced on top afterwards, may be empty, e.g. "pydantic>=2,<3"
#   TEST_CMD     pytest invocation, e.g. "pytest tests"
# Outputs in /out: install.log, freeze.txt, pytest.log, junit.xml
# Exit code: 10 = install failed, 11 = upgrade failed, otherwise pytest's own exit code.
set -u

# /src is the host checkout, mounted read-only. Work on a copy so nothing the
# install or tests write (egg-info, caches, temp files) leaks back to the host.
cp -a /src/. /work/
cd /work

sh -c "$INSTALL_CMD" > /out/install.log 2>&1 || exit 10

# Install the repo first (its own pins pull in the old version), then force the
# new version on top. This mimics "bump the dependency" without editing the repo.
if [ -n "${UPGRADE:-}" ]; then
    # Unquoted on purpose: UPGRADE may hold several requirements.
    pip install --upgrade $UPGRADE >> /out/install.log 2>&1 || exit 11
fi

pip freeze > /out/freeze.txt

sh -c "$TEST_CMD --junitxml=/out/junit.xml -p no:cacheprovider" > /out/pytest.log 2>&1
