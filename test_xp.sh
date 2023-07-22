#!/bin/bash

python ./main.py -e $1/env_params -p $1/model_params -m test $1/best_model.pt $2
