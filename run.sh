#!/bin/bash

echo 'starting training!'
python mnist_cnn.py

echo 'fine tuning and creating submission!'
python create_submission.py
