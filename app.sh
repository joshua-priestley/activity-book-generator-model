#!/bin/sh

export FLASK_APP=words-api
export FLASK_ENV=development

python -m flask run --host=0.0.0.0
