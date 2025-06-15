#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TODAY=$(date '+%d-%m-%Y_%H:%M:%S')
# echo "SCRIPT_DIR: $SCRIPT_DIR"

source_or_setup_venv() {
    local venv_path="$1"
    local index="$2"
    
    if [ -d "$venv_path" ]; then
        source "$venv_path/bin/activate"
        echo "$venv_path/bin/activate: Succesfully activated."
    else
        echo "$venv_path/bin/activate: Virtual environment not found"
        virtualenv "$venv_path"
        source "$venv_path/bin/activate"
        pip install -r ${arr3[$index]}
        echo "$venv_path/bin/activate Virtual environment installed!"
    fi
    }

declare -a arr=(
"$SCRIPT_DIR/venv/preprocess" 
"$SCRIPT_DIR/venv/ocr"
"$SCRIPT_DIR/venv/postprocess"
"$SCRIPT_DIR/venv/evaluate")

declare -a arr2=(
"$SCRIPT_DIR/src/preprocess/preprocess.py" 
"$SCRIPT_DIR/src/ocr/ocr.py"
"$SCRIPT_DIR/src/postprocess/postprocess.py"
"$SCRIPT_DIR/src/evaluate/evaluate.py")

declare -a arr3=(
"$SCRIPT_DIR/requirements/preprocess_requirements.txt" 
"$SCRIPT_DIR/requirements/ocr_requirements.txt"
"$SCRIPT_DIR/requirements/postprocess_requirements.txt"
"$SCRIPT_DIR/requirements/evaluate_requirements.txt")

for i in "${arr[@]}"
do
    # echo "Index $index, Value: $i"
    source_or_setup_venv "$i" "$index"

    BASE=$(basename "${arr[$index]}")
    LOG_FILE="$SCRIPT_DIR/logs/"$BASE"_$TODAY.out"

    python3 -u "${arr2[$index]}" >> "$LOG_FILE" 2>&1

    ((index++))
    deactivate 
    echo "$i: Succesfully deactivated."
done

echo "Pipeline done!"

