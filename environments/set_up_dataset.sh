#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"

cd "$ROOT_DIR"
export HF_TOKEN="$(python environments/get_hf_token.py)"

mkdir -p "$DATA_DIR"

if [[ ! -d "$DATA_DIR/MangaSegmentation" ]]; then
    hf download MS92/MangaSegmentation --repo-type=dataset --local-dir "$DATA_DIR/MangaSegmentation" --token "$HF_TOKEN"
    rm -rf "$DATA_DIR/MangaSegmentation/.cache"
else
    echo "$DATA_DIR/MangaSegmentation already exists, skipping download."
fi

if [[ ! -d "$DATA_DIR/MangaSegmentation/jsons_processed" ]]; then
    gdown --fuzzy "https://drive.google.com/file/d/1IvwKFzQgSkDpJQT69wpHlChkhBtfm_p3/view?usp=sharing" -O "$DATA_DIR/MangaSegmentation/jsons_processed.zip"
    unzip -o "$DATA_DIR/MangaSegmentation/jsons_processed.zip" -d "$DATA_DIR/MangaSegmentation/"
    rm "$DATA_DIR/MangaSegmentation/jsons_processed.zip"
else
    echo "$DATA_DIR/MangaSegmentation/jsons_processed already exists, skipping download."
fi

if [[ ! -d "$DATA_DIR/jesc-2019" ]]; then
    mkdir -p "$DATA_DIR/jesc-2019"
    gdown --fuzzy "https://drive.google.com/file/d/1_N--o0k14rjb5aQKZr12Isf2YL9DNr2f/view?usp=sharing" -O "$DATA_DIR/jesc-2019/dataset.csv"
else
    echo "$DATA_DIR/jesc-2019 already exists, skipping download."
fi

if [[ ! -d "$DATA_DIR/Manga109_released_2023_12_07" ]]; then
    hf download hal-utokyo/Manga109 --repo-type dataset --include Manga109_released_2023_12_07.zip --local-dir "$DATA_DIR" --token "$HF_TOKEN"
    unzip "$DATA_DIR/Manga109_released_2023_12_07.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/Manga109_released_2023_12_07.zip"
    rm -rf "$DATA_DIR/.cache" "$DATA_DIR/__MACOSX"
else
    echo "$DATA_DIR/Manga109_released_2023_12_07 already exists, skipping download."
fi

if [[ ! -d "$DATA_DIR/open-mantra-dataset" ]]; then
    git clone https://github.com/mantra-inc/open-mantra-dataset.git "$DATA_DIR/open-mantra-dataset"
    rm -rf "$DATA_DIR/open-mantra-dataset/.git/"
else
    echo "$DATA_DIR/open-mantra-dataset already exists, skipping download."
fi
