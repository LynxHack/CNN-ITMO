#!/bin/bash

set -e

coverage erase
make clean
make install-coverage

python selftest.py
python -m pytest -vx --cov PIL --cov-report term Tests

pushd /tmp/check-manifest && check-manifest --ignore ".coveragerc,.editorconfig,*.yml,*.yaml,tox.ini" && popd

# Docs
if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ]; then make doccheck; fi
