#!/bin/bash

echo "Cleaning up all artifacts from the workflow"
echo ""

echo "Clearing *.log"
rm -f *.log
rm -f src/*.log
rm -f artifacts/*.log

clear_dir() {
    if [ -d "$1" ]; then
        echo "Clearing $1/*"
        find "$1" -mindepth 1 -delete
    else
        echo "Directory $1 does not exist, skipping."
    fi
}

clear_dir "artifacts/runs"
clear_dir "artifacts/replays"
clear_dir "artifacts/checkpoints"

echo "Done."
