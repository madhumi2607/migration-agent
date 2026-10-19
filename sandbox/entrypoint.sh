#!/bin/sh
# Runs inside the sandbox container. Inputs (env vars, set by the sandbox server):
#   INSTALL_CMD  installs the repo with its OWN dependencies, e.g. "pip install -e ."
#                The agent controls these (via the repo's pyproject/requirements).
#   TEST_DEPS    harness-chosen test tools and era pins, may be empty,
#                e.g. 'pytest "shapely>=2.0,<2.1"' (shell-quoted requirements)
#   UPGRADE      pip requirement(s) forced on top afterwards, may be empty, e.g. "pydantic>=2,<3"
#   TEST_CMD     pytest invocation, e.g. "pytest tests"
# Outputs in /out: install.log, freeze.txt, pytest.log, junit.xml
# Exit code: 10 = install failed, 11 = upgrade failed, otherwise pytest's own exit code.
set -u

# /src is the host checkout, mounted read-only. Work on a copy so nothing the
# install or tests write (egg-info, caches, temp files) leaks back to the host.
cp -a /src/. /work/
cd /work

# 1. The repo and its own pins. Kept separate from the harness's pins so that when the
#    agent bumps a dependency (correctly), the harness can never make this step fail.
sh -c "$INSTALL_CMD" > /out/install.log 2>&1 || exit 10

# 2. Test tools and era pins. A separate pip call only warns (never fails) if a pin here
#    disagrees with the repo's own requirements; step 3 then moves things forward anyway.
if [ -n "${TEST_DEPS:-}" ]; then
    sh -c "pip install $TEST_DEPS" >> /out/install.log 2>&1 || exit 10
fi

# 3. Force the new version on top. This mimics "bump the dependency" without editing the repo.
if [ -n "${UPGRADE:-}" ]; then
    # Unquoted on purpose: UPGRADE may hold several requirements.
    pip install --upgrade $UPGRADE >> /out/install.log 2>&1 || exit 11
fi

pip freeze > /out/freeze.txt

# xunit1 adds file= and line= to each <testcase>, which lets the report build
# runnable test ids (tests/x.py::TestA::test_b) instead of dotted class names.
# --continue-on-collection-errors: by default one test module that fails to import
# aborts the whole run, hiding every other failure from the agent.
sh -c "$TEST_CMD --junitxml=/out/junit.xml -o junit_family=xunit1 --continue-on-collection-errors -p no:cacheprovider" > /out/pytest.log 2>&1
