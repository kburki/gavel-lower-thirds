#!/bin/bash
# Debug script to visualize lower thirds extraction results

# Get the directory of this script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Default values
JSON_PATH=""
VIDEO_PATH=""
OUTPUT_DIR="$SCRIPT_DIR/output/reports"

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
    echo "Usage: $0 [--json JSON_FILE] [--video VIDEO_FILE] [--output OUTPUT_DIR]"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Create a simple HTML report directly using shell commands
echo "Generating a basic HTML report for debugging..."

# Load the JSON file
JSON_CONTENT=$(cat "$JSON_PATH")

# Extract the video name from JSON
VIDEO_NAME=$(echo $JSON_CONTENT | grep -o '"video": *"[^"]*"' | sed 's/"video": *"//;s/"$//')

# Count the number of lower thirds
COUNT=$(echo $JSON_CONTENT | grep -o '"lower_thirds": *\[' | wc -l)

# Generate a simple HTML report
HTML_PATH="$OUTPUT_DIR/debug_report.html"
echo "<!DOCTYPE html>
<html>
<head>
    <title>Lower Thirds Debug Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #004165; }
        pre { background-color: #f5f5f5; padding: 10px; overflow: auto; }
    </style>
</head>
<body>
    <h1>Lower Thirds Debug Report</h1>
    <p>Generated on $(date)</p>
    <p>JSON File: $JSON_PATH</p>
    <p>Video: $VIDEO_NAME</p>
    
    <h2>First 10 Lower Thirds:</h2>
    <pre>" > $HTML_PATH

# Extract and display the first 10 lower thirds
echo $JSON_CONTENT | grep -o '"lower_thirds": *\[.*\]' | sed 's/"lower_thirds": *//' | 
    python3 -c "
import sys, json
data = json.load(sys.stdin)
for i, lt in enumerate(data[:10]):
    print(f\"#{i+1}: {lt.get('type', 'Unknown')} - {lt.get('main_text', '')}\\n  {lt.get('context_text', '')}\\n  {lt.get('start_time_str', '')} to {lt.get('end_time_str', '')}\\n\")
" >> $HTML_PATH

echo "</pre>

<h2>Full JSON Content:</h2>
<pre>
$(echo $JSON_CONTENT | python3 -m json.tool)
</pre>

</body>
</html>" >> $HTML_PATH

echo "Debug report created at: $HTML_PATH"

# Now try to run the actual visualizer
echo "Attempting to run the visualizer..."

if [ ! -z "$VIDEO_PATH" ]; then
    VISUALIZE_CMD="python3 $SCRIPT_DIR/visualize_lower_thirds.py \"$JSON_PATH\" --output \"$OUTPUT_DIR\" --video \"$VIDEO_PATH\""
else
    VISUALIZE_CMD="python3 $SCRIPT_DIR/visualize_lower_thirds.py \"$JSON_PATH\" --output \"$OUTPUT_DIR\""
fi

echo "Running command: $VISUALIZE_CMD"
eval $VISUALIZE_CMD
RESULT=$?

echo "Visualizer exit code: $RESULT"

# Check if the HTML report was created
REPORT_PATH="$OUTPUT_DIR/lower_thirds_report.html"
if [ -f "$REPORT_PATH" ]; then
    echo "Report successfully created at: $REPORT_PATH"
else
    echo "No report file found at: $REPORT_PATH"
    echo "Checking if any HTML files were created:"
    find "$OUTPUT_DIR" -name "*.html"
fi

# Deactivate virtual environment if it was activated
if [ -d "$SCRIPT_DIR/venv" ]; then
    deactivate
fi