#!/bin/bash

#download weights for nets

OUTPUT_PATH=/app/work_dir/pretrained
mkdir -p ${OUTPUT_PATH}

#for detection
DETECTION_NET='1-G4JUK4adVH7OpfGw6ycxZn5cCDAna_5'
gdown --id ${DETECTION_NET} --fuzzy --output ${OUTPUT_PATH}/detection.pth

#for colors
CLASSIFICATION_NET='1-jK92IIoPk8YKqYdYEqFhyePRrV8-GBO'
gdown --id ${COLOR_NET} --fuzzy --output ${OUTPUT_PATH}/classification.pth
