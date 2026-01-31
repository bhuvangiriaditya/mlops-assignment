#!/bin/bash
set -e

mkdir -p data/raw
cd data/raw

# Check if data exists
if [ -d "Cat" ] && [ -d "Dog" ]; then
    echo "Data already exists. Skipping download."
    exit 0
fi

echo "Checking for Kaggle API credentials..."
if [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f ./kaggle.json ]; then
    echo "Standard download requires Kaggle API keys."
    echo "Attempting to use direct 'opendatasets' style or asking user."
    echo "For this environment, please ensure ~/.kaggle/kaggle.json exists."
    # Fallback: Try to use the previous direct link if Kaggle fails? 
    # No, user explicitly asked for this dataset.
    
    # We will try to use the Kaggle CLI assuming it's configured or environment variables are set.
fi

echo "Downloading bhavikjikadara/dog-and-cat-classification-dataset..."
# Ensure kaggle is installed
pip install kaggle --quiet

# Download
kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset --unzip

echo "Organizing..."
# Dataset likely extracts to current dir or a subfolder. 
# Based on structure: usually creates 'Dog' and 'Cat' folders directly or in a parent.
# We will check.

if [ -d "Dog" ] && [ -d "Cat" ]; then
    echo "Found Dog and Cat folders."
    mv Dog dogs
    mv Cat cats
elif [ -d "dataset/Dog" ]; then
    mv dataset/Dog dogs
    mv dataset/Cat cats
    rm -r dataset
fi

# Cleanup
rm -f dog-and-cat-classification-dataset.zip

echo "Removing corrupt files..."
find . -name "*.jpg" -size 0 -delete

echo "Dataset setup complete in data/raw"
