#!/usr/bin/env bash

git clone -b my-branch git@github.com:user/myproject.git

if [ ! -f training.csv ]; then
    aws s3 cp s3://fakenerd/training.csv .
fi
if [ ! -f test.csv ]; then
    aws s3 cp s3://fakenerd/test.csv .
fi

source activate kaggle-facial-keypoint-detection
python run_in_ec2.py

aws s3 sync test_checkpoints s3://fakenerd/test_checkpoints
aws s3 sync test_summaries s3://fakenerd/test_summaries
