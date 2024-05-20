#!/bin/bash

set -x

# seems protobufs package is too new, also lazy eval has issues
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /books/MachineLearning/GenerativeDeepLearning/venv/bin/tensorboard --logdir logs --load_fast=false
