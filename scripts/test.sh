#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
export PYTHONPATH=./
PYTHON=python
TEST_CODE=test.py
CONFIG_PATH="None"
SAVE_PATH="log/wcs3d"
WEIGHT_PATH="weight/model_best_wcs3d.pth"

while getopts "p:r:c:s:w:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      TEST_CODE=$OPTARG
      ;;
    c)
      CONFIG_PATH=$OPTARG
      ;;
    s)
      SAVE_PATH=$OPTARG
      ;;
    w)
      WEIGHT_PATH=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

echo "Python interpreter dir: $PYTHON"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SAVE_PATH: $SAVE_PATH"
echo "WEIGHT_PATH: $WEIGHT_PATH"

echo " =========> RUN TASK <========="

#$PYTHON -u "$CODE_DIR"/tools/$TEST_CODE \
$PYTHON -u tools/$TEST_CODE \
  --config "$CONFIG_PATH" \
  --options save_path="$SAVE_PATH" weight="$WEIGHT_PATH"
