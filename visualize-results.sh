#!/bin/bash
# Script to visualize lower thirds extraction results

# Get the directory of this script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Default values
JSON_PATH=""
VIDEO_PATH=""
OUTPUT_DIR="$SCRIPT_DIR/output/reports"
NO_THUMBNAILS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json|-j)
            JSON_PATH="$2"
            shift 2
            ;;
        --video|-v)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-thumbnails|-n)
            NO_THUMBNAILS=true
            shift
            ;;
        *)
            # If no flag, assume it's the JSON path
            if [ -z "$JSON_PATH" ]; then
                JSON_PATH="$1"
            elif [ -z "$VIDEO_PATH" ]; then
                VIDEO_PATH="$1"
            fi
            shift
            ;;
    esac
done

# Check if JSON path is provided
if [ -z "$JSON_PATH" ]; then
    echo "Error: No JSON file path provided"
    echo "Usage: $0 [--json JSON_FILE] [--video VIDEO_FILE] [--output OUTPUT_DIR] [--no-thumbnails]"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Build command
echo "JSON Path: $JSON_PATH"
if [ ! -z "$VIDEO_PATH" ]; then
    echo "Video Path: $VIDEO_PATH"
    # Check if video exists
    if [ ! -f "$VIDEO_PATH" ]; then
        echo "Warning: Video file does not exist at $VIDEO_PATH"
        # Try to find it in common locations
        if [ -f "./video/$(basename "$VIDEO_PATH")" ]; then
            VIDEO_PATH="./video/$(basename "$VIDEO_PATH")"
            echo "Found video at: $VIDEO_PATH"
        elif [ -f "../video/$(basename "$VIDEO_PATH")" ]; then
            VIDEO_PATH="../video/$(basename "$VIDEO_PATH")"
            echo "Found video at: $VIDEO_PATH"
        fi
    fi
fi
echo "Output Directory: $OUTPUT_DIR"

CMD="python3 $SCRIPT_DIR/visualize_lower_thirds.py \"$JSON_PATH\" --output \"$OUTPUT_DIR\""

if [ ! -z "$VIDEO_PATH" ]; then
    CMD="$CMD --video \"$VIDEO_PATH\""
fi

if [ "$NO_THUMBNAILS" = true ]; then
    CMD="$CMD --no-thumbnails"
fi

# Run the visualization
echo "Generating visualization report..."
echo "Running command: $CMD"
eval $CMD
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "Report generated successfully."
    
    # Get the HTML file path
    REPORT_PATH="$OUTPUT_DIR/lower_thirds_report.html"
    
    if [ -f "$REPORT_PATH" ]; then
        echo "Report created at: $REPORT_PATH"
        
        # Try to open the report in the default browser
        if command -v xdg-open &> /dev/null; then
            echo "Opening report in browser..."
            xdg-open "$REPORT_PATH"
        elif command -v open &> /dev/null; then
            echo "Opening report in browser..."
            open "$REPORT_PATH"
        else
            echo "To view the report, open this file in your browser: $REPORT_PATH"
        fi
    else
        echo "Report file not found at expected location: $REPORT_PATH"
        echo "Checking for other HTML files in output directory:"
        find "$OUTPUT_DIR" -name "*.html"
    fi
else
    echo "Error generating report. Exit code: $RESULT"
    
    # Try to create a simple fallback report
    echo "Creating a simple fallback report..."
    python3 "$SCRIPT_DIR/simple_visualizer.py" "$JSON_PATH" "$OUTPUT_DIR"
fi

# Deactivate virtual environment if it was activated
if [ -d "$SCRIPT_DIR/venv" ]; then
    deactivate
fi