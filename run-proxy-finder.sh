#!/bin/bash
# Run the proxy finder to scan shares and process Gavel Alaska videos

# Get the directory of this script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Default values
CONFIG_PATH="$SCRIPT_DIR/config/config.json"
PROCESS=false
LIST_ONLY=false
LIMIT=""
SINGLE=false
VERBOSE=false

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo "Scan SNS EVO shares for Gavel Alaska videos and process them"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE    Config file path (default: ./config/config.json)"
    echo "  -p, --process        Process the found files"
    echo "  -l, --list-only      Only list matching files without processing"
    echo "  -n, --limit N        Limit processing to N files (for testing)"
    echo "  -s, --single         Process only the first file (for testing)"
    echo "  -v, --verbose        Enable verbose logging"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --process                     Process all matching files"
    echo "  $0 --list-only                   List all matching files"
    echo "  $0 --process --single            Process only the first file (test mode)"
    echo "  $0 --process --limit 5           Process only 5 files"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -p|--process)
            PROCESS=true
            shift
            ;;
        -l|--list-only)
            LIST_ONLY=true
            shift
            ;;
        -n|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -s|--single)
            SINGLE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Build command arguments
ARGS=""
ARGS="$ARGS --config \"$CONFIG_PATH\""

if [ "$PROCESS" = true ]; then
    ARGS="$ARGS --process"
fi

if [ "$LIST_ONLY" = true ]; then
    ARGS="$ARGS --list-only"
fi

if [ -n "$LIMIT" ]; then
    ARGS="$ARGS --limit $LIMIT"
fi

if [ "$SINGLE" = true ]; then
    ARGS="$ARGS --single"
fi

if [ "$VERBOSE" = true ]; then
    ARGS="$ARGS --verbose"
fi

# Create necessary directories
mkdir -p "$SCRIPT_DIR/output/json"
mkdir -p "$SCRIPT_DIR/output/csv"
mkdir -p "$SCRIPT_DIR/temp"

# Run the proxy finder
echo "Running EVO proxy finder..."
eval "python $SCRIPT_DIR/evo_proxy_finder.py $ARGS"

# Deactivate virtual environment if it was activated
if [ -d "$SCRIPT_DIR/venv" ]; then
    deactivate
fi