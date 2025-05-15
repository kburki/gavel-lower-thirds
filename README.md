### Testing and Limited Processing

To test the system without processing all files:

```bash
# Process only the first file found (best for initial testing)
./run-proxy-finder.sh --process --single

# Process only the first 5 files
./run-proxy-finder.sh --process --limit 5

# List files first, then process a single file
./run-proxy-finder.sh --list-only
./run-proxy-finder.sh --process --single
```

For slow connections or VPNs, the system processes files sequentially by default to avoid bandwidth issues.## EVO Integration with Proxy Detection

The tool also includes a proxy finder that can scan your SNS EVO shares for videos matching the Gavel Alaska naming patterns. This allows you to automatically process all matching videos using their proxy files.

### Proxy Finder Features

- **Automatic Share Scanning**: Scans configured EVO shares for videos matching Gavel Alaska naming patterns
- **Proxy Retrieval**: Gets proxy URLs from ShareBrowser's API
- **Parallel Processing**: Processes multiple videos simultaneously for better performance
- **Temporary Storage**: Downloads proxies to a temporary location for processing

### Proxy File Patterns

The tool looks for files matching these patterns:
- `G[SHJOG][A-Z]{2,4}\d{6}[A-Z]?` (e.g., GSFIN230415A)
- `AKSC\d{6}[A-Z]?` (e.g., AKSC230415A)

### Running the Proxy Finder

```bash
# List matching files in all configured shares
./run-proxy-finder.sh --list-only

# Process all matching files
./run-proxy-finder.sh --process

# Use a specific configuration file
./run-proxy-finder.sh --config custom-config.json --process
```

### Share Configuration

Configure the shares to monitor in your `config.json`:

```json
"evo_shares": [
  { 
    "name": "sns3-Gavel_25-1", 
    "path": "/",
    "volumeUuid": "02841c18-ec66-4d1f-b7c2-c4081ac86c2e"
  },
  { 
    "name": "sns5-Gavel25_2", 
    "path": "/",
    "volumeUuid": "01001a4b-3e4a-4eae-9b74-e0a1e2523477"
  }
]
```# Gavel Alaska Lower Thirds Extractor

A specialized tool for extracting lower thirds text from Gavel Alaska legislative videos and integrating with SNS EVO shared storage systems.

## Overview

This tool analyzes video files to detect and extract text from lower thirds graphics (chyrons), which typically identify speakers, bills, events, and other information in legislative footage. The extracted information can be used for searching, cataloging, and research purposes.

## Features

- **Automatic Lower Thirds Detection**: Identifies the presence of lower thirds graphics in video frames
- **OCR Text Extraction**: Uses Tesseract OCR to extract text from the detected graphics
- **Metadata Generation**: Creates structured metadata from the extracted text
- **EVO Integration**: Uploads metadata as tags to SNS EVO shared storage systems via the Slingshot API
- **Database Storage**: Optional database storage for more advanced querying and analysis
- **CSV Export**: Export results to CSV files for further analysis
- **Debug Mode**: Saves intermediate frames for troubleshooting

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- Pytesseract
- Tesseract OCR engine

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gavel-alaska-lower-thirds-extractor.git
   cd gavel-alaska-lower-thirds-extractor
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. Configure the tool by creating a `config.json` file (see Configuration section below)

## Usage

### Basic Usage

```bash
# Process a single video file
python lower_thirds_extractor.py /path/to/video.mp4 --output ./output

# Process a directory of videos
python lower_thirds_extractor.py /path/to/videos_directory --output ./output

# Process and upload to EVO
python lower_thirds_extractor.py /path/to/video.mp4 --output ./output --upload

# Enable debug mode to save frames
python lower_thirds_extractor.py /path/to/video.mp4 --debug

# Export results to CSV
python lower_thirds_extractor.py /path/to/video.mp4 --csv

# Store results in a database
python lower_thirds_extractor.py /path/to/video.mp4 --db
```

### Command Line Arguments

- `video_path`: Path to the video file or directory of videos
- `--output, -o`: Output directory for metadata files
- `--config, -c`: Path to configuration JSON file
- `--upload, -u`: Upload metadata to EVO
- `--debug, -d`: Enable debug mode with image saves
- `--csv`: Export data to CSV
- `--db`: Store data in SQLite database

## Configuration

Create a `config.json` file in the project directory or specify a path with the `--config` option. Here's an example configuration:

```json
{
    "evo_settings": {
        "evo_address": "http://your-evo-server",
        "username": "your-username",
        "password": "your-password"
    },
    
    "video_settings": {
        "sampling_rate": 1,
        "min_text_confidence": 60,
        "tesseract_path": "/usr/local/bin/tesseract"
    },
    
    "lower_third_detection": {
        "blue_hsv_lower": [90, 50, 50],
        "blue_hsv_upper": [130, 255, 255],
        "blue_density_threshold": 0.15,
        "edge_density_threshold": 0.1
    },
    
    "regions": {
        "modern_widescreen": [
            {"name": "full_lower_third", "y1": 0.75, "y2": 1.0, "x1": 0, "x2": 1.0},
            {"name": "top_bar", "y1": 0.75, "y2": 0.85, "x1": 0, "x2": 1.0},
            {"name": "bottom_bar", "y1": 0.85, "y2": 1.0, "x1": 0, "x2": 1.0}
        ],
        "legacy_4x3": [
            {"name": "full_lower_third", "y1": 0.75, "y2": 1.0, "x1": 0, "x2": 1.0},
            {"name": "top_bar", "y1": 0.75, "y2": 0.85, "x1": 0, "x2": 1.0},
            {"name": "bottom_bar", "y1": 0.85, "y2": 1.0, "x1": 0, "x2": 1.0}
        ]
    },
    
    "markers": {
        "speaker": ["Sen.", "Rep.", "Speaker", "President", "Chairman", "Chair"],
        "bill": ["HB", "SB", "SCR", "HCR", "Bill", "CS", "Resolution", "Amendment", "Motion"],
        "event": ["Presentation", "Pledge", "Introduction", "At Ease", "Colors", "Allegiance", "Recess", "Adjournment"],
        "voice": ["Voice of:"],
        "next": ["Next"]
    },
    
    "ocr_settings": {
        "config": "--oem 1 --psm 6 -l eng --dpi 300",
        "preprocessor": "adaptive_threshold"
    },
    
    "database_settings": {
        "use_database": true,
        "database_path": "lower_thirds.db"
    },
    
    "debug_mode": false,
    
    "output_settings": {
        "default_output_dir": "./output",
        "export_csv": true
    }
}
```

### Security Note

The configuration file contains sensitive information like EVO credentials. Make sure to:
- Keep it out of version control (add to `.gitignore`)
- Set appropriate file permissions
- Consider using environment variables for credentials in production environments

## Output

The tool produces several types of output:

### JSON Metadata

For each processed video, a JSON file is created containing all extracted lower thirds with timestamps:

```json
{
  "video": "senate_session.mp4",
  "extraction_date": "2025-05-15T10:30:45.123456",
  "lower_thirds": [
    {
      "main_text": "Sen. Robert Yundt",
      "context_text": "Senate Floor Session KTOO Tue. 1/21/25 1:48 pm",
      "type": "Speaker",
      "start_time": 2961.0,
      "end_time": 3001.3,
      "start_time_str": "00:49:21",
      "end_time_str": "00:50:01",
      "duration": 40.3,
      "date": "Tue. 1/21/25",
      "time": "1:48 pm",
      "session": "Senate Floor"
    },
    // More lower thirds...
  ]
}
```

### EVO Tags

When uploading to EVO, the following tag categories are generated:

- `Speaker:Name` - For identified speakers
- `Bill:Number` - For bill references
- `Event:Name` - For special events
- `Voice:Name` - For "Voice of" attributions
- `Time:HH:MM:SS` - Timestamps
- `Date:MM/DD/YY` - Extracted dates
- `Session:Type` - Session information
- `Organization:Name` - For identified organizations

### Database Storage

When using database storage, three tables are created:

1. `videos` - Information about processed videos
2. `lower_thirds` - All extracted lower thirds with timestamps
3. `tags` - Generated tags linked to lower thirds

This enables complex queries and analysis across multiple videos.

### CSV Export

Two CSV files are generated per video:

1. `video_name_lower_thirds.csv` - All extracted lower thirds
2. `video_name_tags.csv` - All generated tags

## Custom Development

### Extending the Tool

You can extend the tool's capabilities by:

1. **Adding New Lower Third Types**: Modify the `markers` section in the configuration and update the `_determine_lower_third_type` method
2. **Customizing Tag Generation**: Modify the `_generate_tags_for_lower_third` method
3. **Enhancing Detection**: Tune the detection parameters in the configuration

## Troubleshooting

If the tool is not detecting lower thirds correctly:

1. Enable debug mode (`--debug`) to see what frames are being analyzed
2. Adjust the detection parameters in the configuration
3. Check the log file for errors and warnings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text recognition
- [OpenCV](https://opencv.org/) for image processing
- [Gavel Alaska](https://www.ktoo.org/gavel/) for providing legislative video coverage
- [Studio Network Solutions](https://www.studionetworksolutions.com/) for EVO shared storage