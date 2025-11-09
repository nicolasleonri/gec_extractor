#!/bin/bash

RESULTS_DIR="./results/csv/multi_label"
OUTPUT_CSV="./results/csv/multi_label/aggregated_results.csv"

# Remove existing CSV if it exists
[ -f "$OUTPUT_CSV" ] && rm "$OUTPUT_CSV"

# Header row
echo "prefix,prefix2,augmentation,ht,cv,samples,metric,date,eval_accuracy,eval_mean_task_accuracy,eval_macro_f1" > "$OUTPUT_CSV"

# Loop through all JSON files
for filepath in "$RESULTS_DIR"/*.json; do
    filename=$(basename "$filepath")
    basename_no_ext="${filename%.json}"

    # Parse filename from the right-hand side
    date="${basename_no_ext##*_}"
    basename_no_date="${basename_no_ext%_*}"

    metric="${basename_no_date##*_}"
    basename_no_metric="${basename_no_date%_*}"

    samples="${basename_no_metric##*_}"
    basename_no_samples="${basename_no_metric%_*}"

    cv="${basename_no_samples##*_}"
    basename_no_cv="${basename_no_samples%_*}"

    ht="${basename_no_cv##*_}"
    basename_no_ht="${basename_no_cv%_*}"

    prefix_and_aug="${basename_no_ht}"
    IFS='_' read -r prefix prefix2 augmentation <<< "$prefix_and_aug"

    # Extract only the desired JSON values
    eval_accuracy=$(grep -oP '"eval_accuracy":\s*\K[0-9.]+|null' "$filepath")
    # eval_mean_task_accuracy=$(grep -oP '"eval_mean_task_accuracy":\s*\K[0-9.]+|null' "$filepath")
    eval_macro_f1=$(grep -oP '"eval_macro_f1":\s*\K[0-9.]+|null' "$filepath")

    # Append row to CSV
    echo "$prefix,$prefix2,$augmentation,$ht,$cv,$samples,$metric,$date,$eval_accuracy,$eval_mean_task_accuracy,$eval_macro_f1" >> "$OUTPUT_CSV"
done

echo "✅ Aggregated JSON files into $OUTPUT_CSV"
