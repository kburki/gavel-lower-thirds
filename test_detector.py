#!/usr/bin/env python3
"""
Test script for the Lower Thirds Detector

Tests detection on still images to verify the algorithm works correctly.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import logging
from pathlib import Path

from lower_thirds import LowerThirdsDetector
from config_manager import ConfigManager
from utils import setup_logging, save_debug_image

def main():
    """Main entry point for the test script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test lower thirds detection on images')
    parser.add_argument('image_path', help='Path to the image file or directory of images')
    parser.add_argument('--output', '-o', default='debug_frames', help='Output directory for results')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with detailed output')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(debug=args.debug)
    
    # Load configuration
    logger.info("Loading configuration...")
    config_manager = ConfigManager(args.config)
    config_manager.update('debug_mode', True)  # Always enable debug for the test
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process image(s)
    if os.path.isdir(args.image_path):
        # Process all images in directory
        image_files = []
        for filename in os.listdir(args.image_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(args.image_path, filename))
        
        if not image_files:
            logger.error(f"No image files found in directory: {args.image_path}")
            return
        
        logger.info(f"Processing {len(image_files)} image files...")
        
        # Process each image
        for image_path in image_files:
            test_image(image_path, args.output, config_manager, logger)
    else:
        # Process single image
        test_image(args.image_path, args.output, config_manager, logger)
    
    logger.info("Testing complete")

def test_image(image_path, output_dir, config_manager, logger):
    """Test lower thirds detection on a single image"""
    logger.info(f"Processing image: {image_path}")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return
    
    # Get the filename without extension
    base_name = Path(image_path).stem
    
    # Save the original image to output dir
    original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
    cv2.imwrite(original_path, image)
    
    # Create the detector with our configuration
    detector = LowerThirdsDetector(config_manager.get_all())
    
    # Extract the lower third region from the image
    h, w = image.shape[:2]
    lower_region = image[int(h * 0.75):h, 0:w]
    lower_region_path = os.path.join(output_dir, f"{base_name}_lower_region.jpg")
    cv2.imwrite(lower_region_path, lower_region)
    
    # Test lower third detection
    detection_result = detector._is_lower_third(lower_region, 0)
    logger.info(f"Detection result: {'DETECTED' if detection_result else 'NOT DETECTED'}")
    
    # If detected, run OCR
    if detection_result:
        # Create HSV mask for visualization
        hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)
        lower_blue = np.array(config_manager.get('lower_third_detection.blue_hsv_lower', [90, 50, 50]))
        upper_blue = np.array(config_manager.get('lower_third_detection.blue_hsv_upper', [130, 255, 255]))
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_path = os.path.join(output_dir, f"{base_name}_blue_mask.jpg")
        cv2.imwrite(mask_path, blue_mask)
        
        # Preprocess for OCR
        preprocessed = detector._preprocess_for_ocr(lower_region)
        preprocessed_path = os.path.join(output_dir, f"{base_name}_preprocessed.jpg")
        cv2.imwrite(preprocessed_path, preprocessed)
        
        # Extract regions
        h_region, w_region = lower_region.shape[:2]
        top_region = lower_region[0:int(h_region * 0.5), 0:w_region]
        bottom_region = lower_region[int(h_region * 0.5):h_region, 0:w_region]
        
        # Save the regions
        top_region_path = os.path.join(output_dir, f"{base_name}_top_region.jpg")
        bottom_region_path = os.path.join(output_dir, f"{base_name}_bottom_region.jpg")
        cv2.imwrite(top_region_path, top_region)
        cv2.imwrite(bottom_region_path, bottom_region)
        
        # Preprocess and run OCR on each region
        top_preprocessed = detector._preprocess_for_ocr(top_region)
        bottom_preprocessed = detector._preprocess_for_ocr(bottom_region)
        
        # Save preprocessed regions
        top_preprocessed_path = os.path.join(output_dir, f"{base_name}_top_preprocessed.jpg")
        bottom_preprocessed_path = os.path.join(output_dir, f"{base_name}_bottom_preprocessed.jpg")
        cv2.imwrite(top_preprocessed_path, top_preprocessed)
        cv2.imwrite(bottom_preprocessed_path, bottom_preprocessed)
        
        # Run OCR on regions
        import pytesseract
        
        # Make sure tesseract path is set
        tesseract_path = config_manager.get('video_settings.tesseract_path', r'/usr/bin/tesseract')
        if os.path.exists('/opt/homebrew/bin/tesseract'):  # Common macOS path with Homebrew
            tesseract_path = r'/opt/homebrew/bin/tesseract'
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # OCR config
        ocr_config = config_manager.get('ocr_settings.config', '--oem 1 --psm 6 -l eng --dpi 300')
        
        # Run OCR
        top_text = pytesseract.image_to_string(top_preprocessed, config=ocr_config).strip()
        bottom_text = pytesseract.image_to_string(bottom_preprocessed, config=ocr_config).strip()
        
        # Log results
        logger.info(f"Top text: {top_text}")
        logger.info(f"Bottom text: {bottom_text}")
        
        # Determine lower third type
        lt_type = detector._determine_lower_third_type(top_text)
        logger.info(f"Detected type: {lt_type}")
        
        # Save results to text file
        results_path = os.path.join(output_dir, f"{base_name}_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Detection result: {'DETECTED' if detection_result else 'NOT DETECTED'}\n")
            f.write(f"Top text: {top_text}\n")
            f.write(f"Bottom text: {bottom_text}\n")
            f.write(f"Detected type: {lt_type}\n")
    
    return detection_result

if __name__ == "__main__":
    main()
