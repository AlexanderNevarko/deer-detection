#!/bin/bash

#download weights for nets

OUTPUT_PATH=/app/work_dir/pretrained
mkdir -p ${OUTPUT_PATH}

#for detection
DETECTION_NET=''
gdown --id ${DETECTION_NET} --fuzzy --output ${OUTPUT_PATH}/detection.pth

#for colors
CLASSIFICATION_NET=''
gdown --id ${COLOR_NET} --fuzzy --output ${OUTPUT_PATH}/classification.pth
