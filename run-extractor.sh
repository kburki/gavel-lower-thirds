#!/bin/bash
# Script to run the lower thirds extractor on a single video file

# Get the directory of this script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Default values
VIDEO_PATH=""
OUTPUT_DIR="$SCRIPT_DIR/output"
CONFIG_PATH="$SCRIPT_DIR/config/config.json"
UPLOAD=false
DEBUG=false
CSV=false
DB=false

# Display help message
function show_help {
    echo "Usage: $0 [options] <video_path>"
    echo "Extract lower thirds from a video file"
    echo ""
    echo "Options:"
    echo "  -o, --output DIR     Output directory (default: ./output)"
    echo "  -c, --config FILE    Config file path (default: ./config/config.json)"
    echo "  -u, --upload         Upload metadata to EVO"
    echo "  -d, --debug          Enable debug mode with image saves"
    echo "  --csv                Export data to CSV"
    echo "  --db                 Store data in SQLite database"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --debug --csv sample.mp4"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -u|--upload)
            UPLOAD=true
            shift
            ;;
        -d|--debug)
            DEBUG=true
            shift
            ;;
        --csv)
            CSV=true
            shift
            ;;
        --db)
            DB=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            # If the argument doesn't start with a dash, treat it as the video path
            if [[ "$1" != -* ]]; then
                VIDEO_PATH="$1"
            else
                echo "Error: Unknown option $1"
                show_help
            fi
            shift
            ;;
    esac
done

# Check if video path is provided
if [ -z "$VIDEO_PATH" ]; then
    echo "Error: No video path provided"
    show_help
fi

# Check if video path exists
if [ ! -e "$VIDEO_PATH" ]; then
    echo "Error: Video file does not exist: $VIDEO_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR/json"
mkdir -p "$OUTPUT_DIR/csv"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Build command arguments
ARGS=""
ARGS="$ARGS --output \"$OUTPUT_DIR\""
ARGS="$ARGS --config \"$CONFIG_PATH\""

if [ "$UPLOAD" = true ]; then
    ARGS="$ARGS --upload"
fi

if [ "$DEBUG" = true ]; then
    ARGS="$ARGS --debug"
fi

if [ "$CSV" = true ]; then
    ARGS="$ARGS --csv"
fi

if [ "$DB" = true ]; then
    ARGS="$ARGS --db"
fi

# Run the extractor
echo "Running Lower Thirds Extractor on $VIDEO_PATH..."
eval "python $SCRIPT_DIR/lower_thirds_extractor.py $ARGS \"$VIDEO_PATH\""

# Deactivate virtual environment if it was activated
if [ -d "$SCRIPT_DIR/venv" ]; then
    deactivate
fi