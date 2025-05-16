#!/usr/bin/env python3
"""
Lower Thirds Extractor for Gavel Alaska

A refactored version using a modular approach:
1. Processes video files (local or proxies from SNS EVO)
2. Detects lower thirds graphics
3. Extracts text using OCR
4. Generates metadata with speaker names, bill information, and timestamps
5. Exports to JSON and CSV formats
6. Visualizes results with HTML reports (optional)

Usage:
    python lower_thirds_extractor.py video_path --output OUTPUT_DIR [OPTIONS]
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Import our modules
from lower_thirds import LowerThirdsDetector
from config_manager import ConfigManager
from utils import setup_logging, save_json, save_csv, generate_output_filename
from visualize_lower_thirds import LowerThirdsVisualizer

def main():
    """Main entry point for the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract lower thirds from video files')
    parser.add_argument('video_path', help='Path to the video file or directory of videos')
    parser.add_argument('--output', '-o', help='Output directory for metadata files')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--upload', '-u', action='store_true', help='Upload metadata to EVO')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with image saves')
    parser.add_argument('--csv', action='store_true', help='Export data to CSV')
    parser.add_argument('--db', action='store_true', help='Store data in SQLite database')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualization report')
    parser.add_argument('--batch', '-b', action='store_true', 
                      help='Process a directory of videos instead of a single file')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(debug=args.debug)
    
    # Load configuration
    logger.info("Loading configuration...")
    config_manager = ConfigManager(args.config)
    
    # Override with command line args
    if args.debug:
        config_manager.update('debug_mode', True)
    if args.csv:
        config_manager.update('export_csv', True)
    if args.db:
        config_manager.update('database_settings.use_database', True)
    if args.upload:
        config_manager.update('upload_to_evo', True)
    
    # Set output directory
    output_dir = args.output or config_manager.get('output_settings.default_output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video(s)
    if args.batch and os.path.isdir(args.video_path):
        # Process all videos in directory
        video_files = []
        for filename in os.listdir(args.video_path):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_files.append(os.path.join(args.video_path, filename))
        
        if not video_files:
            logger.error(f"No video files found in directory: {args.video_path}")
            return
        
        logger.info(f"Processing {len(video_files)} video files...")
        
        # Process each video
        for video_path in video_files:
            process_video(video_path, output_dir, config_manager, args.visualize, logger)
    else:
        # Process single video
        process_video(args.video_path, output_dir, config_manager, args.visualize, logger)
    
    logger.info("Processing complete")

def process_video(video_path, output_dir, config_manager, visualize=False, logger=None):
    """Process a single video file"""
    if logger:
        logger.info(f"Processing video: {video_path}")
    
    # Create the detector with our configuration
    detector = LowerThirdsDetector(config_manager.get_all())
    
    # Process the video
    lower_thirds = detector.process_video(video_path)
    
    if lower_thirds:
        # Save the results to JSON
        json_output_dir = config_manager.get_output_path('json', output_dir)
        json_path = generate_output_filename(video_path, 'json', json_output_dir)
        save_json(lower_thirds, json_path, video_path)
        
        if logger:
            logger.info(f"Saved JSON results to {json_path}")
        
        # Export to CSV if enabled
        if config_manager.get('export_csv', True):
            csv_output_dir = config_manager.get_output_path('csv', output_dir)
            csv_path = generate_output_filename(video_path, 'csv', csv_output_dir)
            save_csv(lower_thirds, csv_path)
            
            if logger:
                logger.info(f"Saved CSV results to {csv_path}")
        
        # Generate visualization if requested
        if visualize:
            visualizer = LowerThirdsVisualizer(config_manager)
            report_output_dir = os.path.join(output_dir, 'reports', Path(video_path).stem)
            html_path = visualizer.create_report(json_path, report_output_dir)
            
            if logger and html_path:
                logger.info(f"Generated visualization report at {html_path}")
        
        # Upload to EVO if requested
        if config_manager.get('upload_to_evo', False):
            if logger:
                logger.info("EVO Upload capability will be implemented in a future update")
            # TODO: Implement EVO upload functionality
            # This will be implemented in a separate pull request
        
        # Store in database if requested
        if config_manager.get('database_settings.use_database', False):
            if logger:
                logger.info("Database storage will be implemented in a future update")
            # TODO: Implement database storage
            # This will be implemented in a separate pull request
        
        return lower_thirds
    else:
        if logger:
            logger.info("No lower thirds data extracted from video")
        return None

if __name__ == "__main__":
    main()
