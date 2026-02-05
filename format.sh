#!/bin/bash

echo "Running isort..."
isort --profile black --line-length 79 src/

echo "Running black..."
black --line-length 79 src/
black --line-length 79 setup.py

pre-commit run --all-files --hook-stage manual
