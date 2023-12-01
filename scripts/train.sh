#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
export PYTHONPATH=./
PYTHON=python
TRAIN_CODE=train.py
CONFIG_PATH="None"
WEIGHT_PATH="None"
GPU="None"
SAVE_PATH="None"



while getopts "p:d:c:w:g:s:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      TRAIN_CODE=$OPTARG
      ;;
    c)
      CONFIG_PATH=$OPTARG
      ;;
    w)
      WEIGHT_PATH=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    s)
      SAVE_PATH=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "SAVE_PATH: $SAVE_PATH"
echo "Python interpreter dir: $PYTHON"
echo "CONFIG: $CONFIG_PATH"
echo "WEIGHT: $WEIGHT_PATH"
echo "GPU Num: $GPU"

echo " =========> RUN TASK <========="

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON -u tools/$TRAIN_CODE \
    --config "$CONFIG_PATH" \
    --num-gpus "$GPU" \
    --options save_path="$SAVE_PATH"
else
    RESUME=true
    $PYTHON -u tools/$TRAIN_CODE \
    --config "$CONFIG_PATH" \
    --num-gpus "$GPU" \
    --options save_path="$SAVE_PATH" resume="$RESUME" weight="$WEIGHT_PATH"
fi