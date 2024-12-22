#!/bin/bash

# Встановлюємо назву папки для віртуального середовища
ENV_DIR="./_env"

# Перевірка, чи існує віртуальне середовище
if [ ! -d "$ENV_DIR" ]; then
    echo "Віртуальне середовище не знайдено. Створюю..."
    python3 -m venv "$ENV_DIR"
    echo "Віртуальне середовище створено в $ENV_DIR"
else
    echo "Віртуальне середовище вже існує в $ENV_DIR"
fi

# Активуємо віртуальне середовище
echo "Активую віртуальне середовище..."
source "$ENV_DIR/bin/activate"

# Встановлення змінної середовища KERAS_HOME
KERAS_HOME_DIR="$(pwd)/datasets"
export KERAS_HOME="$KERAS_HOME_DIR"
echo "Змінна KERAS_HOME встановлена на $KERAS_HOME"

# Перевірка наявності папки для датасетів
if [ ! -d "$KERAS_HOME_DIR" ]; then
    echo "Створюю директорію для датасетів..."
    mkdir -p "$KERAS_HOME_DIR"
    echo "Директорія $KERAS_HOME_DIR створена."
else
    echo "Директорія $KERAS_HOME_DIR вже існує."
fi

# Виведення повідомлення про успіх
echo "Середовище готове. Використовуйте 'source $ENV_DIR/bin/activate' для активації."
