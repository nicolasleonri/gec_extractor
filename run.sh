#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# echo "SCRIPT_DIR: $SCRIPT_DIR"

source_or_setup_venv() {
    local venv_path="$1"
    
    if [ -d "$venv_path" ]; then
        source "$venv_path/bin/activate"
        echo "$venv_path/bin/activate" "was activated."

    else
        echo "$venv_path/bin/activate Virtual environment not found"
        virtualenv "$venv_path"
        source "$venv_path/bin/activate"
        pip install -r $SCRIPT_DIR/requirements.txt
        echo "$venv_path/bin/activate Virtual environment installed!"
    fi
    }

declare -a arr=("$SCRIPT_DIR/venv/preprocess" 
"$SCRIPT_DIR/venv/ocr"
"$SCRIPT_DIR/venv/postprocess"
"$SCRIPT_DIR/venv/evaluate")

declare -a arr2=("$SCRIPT_DIR/src/preprocess/preprocess.py" 
"$SCRIPT_DIR/src/ocr/ocr.py"
"$SCRIPT_DIR/src/postprocess/postprocess.py"
"$SCRIPT_DIR/src/evaluate/evaluate.py")

index = 0
for i in "${arr[@]}"
do
    # echo "Index $index, Value: $i"
    source_or_setup_venv "$i"
    python3 -u ""${arr2[$index]}
    ((index++))
    echo "Passed!"
done

echo "Pipeline done!"

