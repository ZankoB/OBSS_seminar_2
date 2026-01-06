#!/bin/bash

ANNOT_DIR="annotations"
DATA_DIR="data"
OUT_DIR="records"

mkdir -p "$OUT_DIR"

echo ""
echo "=== Converting ASC annotations to WFDB QRS files ==="
echo ""

for asc_file in "$ANNOT_DIR"/*.asc; do
    record=$(basename "$asc_file" .asc)

    echo "Processing record: $record"

    wrann -r "$DATA_DIR/$record" -a qrs < "$asc_file"

    mv "$DATA_DIR/$record.qrs" "$OUT_DIR/"

    echo "  → Created $OUT_DIR/$record.qrs"
done

echo ""
echo "All files converted!"