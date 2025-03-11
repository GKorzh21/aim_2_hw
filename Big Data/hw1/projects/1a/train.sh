#!/bin/bash

# Получение аргументов
PROJECT_NUMBER=$1
DATASET_PATH=$2

# Путь к Python-скрипту (предполагаем, что скрипт находится в той же директории)
SCRIPT_PATH="$(dirname "$0")/train.py"

# Запуск Python-скрипта с переданными аргументами
python3 "$SCRIPT_PATH" "$PROJECT_NUMBER" "$DATASET_PATH"
