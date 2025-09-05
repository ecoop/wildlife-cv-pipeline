#!/usr/bin/env python3
"""
Video Review Assistant for YOLO Model Validation

Extracts frames from videos, runs YOLO detection, and organizes results for human review.
This helps create high-quality positive/negative training samples from video data.

Usage:
    python src/annotation_assistance/video_review_assistant.py --input_dir path/to/videos --output_dir path/to/review --model_path path/to/model.pt

Workflow:
    1. Extracts frames from videos (configurable interval)
    2. Runs YOLO detection on each frame
    3. Sorts frames into detected/no_detection folders
    4. Saves detection metadata for review
    5. Human reviews and sorts into animal/no_object folders for retraining
"""

import cv2
import json
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import shutil
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, frame_interval=30):
    """Extract frames from video at specified interval"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    extracted_frames = []
    frame_count = 0
    
    video_name = Path(video_path).stem
    
    print(f"Processing {video_name}: {total_frames} frames at {fps:.1f} fps")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Create frame filename with timestamp info
            timestamp_seconds = frame_count / fps
            frame_filename = f"{video_name}_frame_{frame_count:06d}_{timestamp_seconds:.1f}s.jpg"
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append({
                'frame_path': str(frame_path),
                'video_source': str(video_path),
                'frame_number': frame_count,
                'timestamp_seconds': timestamp_seconds
            })
        
        frame_count += 1
    
    cap.release()
    return extracted_frames


def run_yolo_on_frames(frame_list, model, confidence_threshold=0.5):
    """Run YOLO detection on extracted frames"""
    results = []
    
    print(f"Running YOLO detection on {len(frame_list)} frames...")
    
    for frame_info in tqdm(frame_list):
        frame_path = frame_info['frame_path']
        
        # Run detection
        detections = model(frame_path, verbose=False)
        
        detection_data = {
            **frame_info,
            'detections': [],
            'has_detections': False
        }
        
        for result in detections:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if confidence >= confidence_threshold:
                        # Get bounding box coordinates (xyxy format)
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # Convert to (x, y, w, h) format
                        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                        
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = model.names.get(class_id, f"class_{class_id}")
                        
                        detection_data['detections'].append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': bbox
                        })
        
        detection_data['has_detections'] = len(detection_data['detections']) > 0
        results.append(detection_data)
    
    return results


def organize_frames_for_review(detection_results, output_dir):
    """Sort frames into detected/no_detection folders for human review"""
    
    detected_dir = output_dir / 'detected'
    no_detection_dir = output_dir / 'no_detection' 
    metadata_dir = output_dir / 'metadata'
    
    detected_dir.mkdir(exist_ok=True)
    no_detection_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    
    detected_count = 0
    no_detection_count = 0
    
    print("Organizing frames for review...")
    
    for result in tqdm(detection_results):
        frame_path = Path(result['frame_path'])
        frame_filename = frame_path.name
        
        if result['has_detections']:
            # Copy to detected folder
            dest_path = detected_dir / frame_filename
            shutil.copy2(frame_path, dest_path)
            detected_count += 1
        else:
            # Copy to no_detection folder
            dest_path = no_detection_dir / frame_filename
            shutil.copy2(frame_path, dest_path)
            no_detection_count += 1
        
        # Save metadata
        metadata_filename = frame_filename.replace('.jpg', '.json')
        metadata_path = metadata_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"Organized frames:")
    print(f"  Detected: {detected_count}")
    print(f"  No detection: {no_detection_count}")
    
    return detected_count, no_detection_count


def create_review_instructions(output_dir, detected_count, no_detection_count):
    """Create instructions file for human review"""
    
    instructions = f"""
# Frame Review Instructions

This directory contains {detected_count + no_detection_count} frames extracted from videos and processed with YOLO detection.

## Directory Structure:
- `detected/` ({detected_count} frames): Frames where YOLO found objects
- `no_detection/` ({no_detection_count} frames): Frames where YOLO found nothing  
- `metadata/` : JSON files with detection details for each frame

## Review Workflow:

### Step 1: Review detected/ folder
- Look at each frame in `detected/`
- For TRUE POSITIVES (real animals): move to `animal/` folder (create if needed)
- For FALSE POSITIVES (not animals): move to `no_object/` folder (create if needed)

### Step 2: Review no_detection/ folder  
- Look at each frame in `no_detection/`
- Most should be correctly identified as having no animals
- For FALSE NEGATIVES (missed animals): move to `animal/` folder
- For TRUE NEGATIVES (correctly no animals): move to `no_object/` folder

### Step 3: Final folders for retraining:
- `animal/` : All frames with real animals (for positive training samples)
- `no_object/` : All frames without animals (for negative training samples)

## Commands to create review folders:
```bash
mkdir -p animal no_object
```

## Tips:
- Check `metadata/*.json` files for detection details and confidence scores
- Pay attention to borderline cases that might indicate model confusion
- Document any patterns you notice in false positives/negatives
"""
    
    with open(output_dir / 'REVIEW_INSTRUCTIONS.md', 'w') as f:
        f.write(instructions)


def main():
    parser = argparse.ArgumentParser(description='Extract frames and run YOLO for human review')
    parser.add_argument('--input_dir', required=True, help='Directory containing video files')
    parser.add_argument('--output_dir', required=True, help='Output directory for review')
    parser.add_argument('--model_path', required=True, help='Path to YOLO model (.pt file)')
    parser.add_argument('--frame_interval', type=int, default=30, 
                       help='Extract every Nth frame (default: 30)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--video_extensions', nargs='+', 
                       default=['.mp4', '.avi', '.mov', '.mkv'],
                       help='Video file extensions to process')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    if not Path(args.model_path).exists():
        print(f"Error: Model file {args.model_path} does not exist")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_frames_dir = output_dir / 'temp_frames'
    temp_frames_dir.mkdir(exist_ok=True)
    
    print(f"Video Review Assistant")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Frame interval: {args.frame_interval}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    # Load YOLO model
    print("Loading YOLO model...")
    try:
        model = YOLO(args.model_path)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find video files
    video_files = []
    for ext in args.video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Extract frames from all videos
    all_frames = []
    for video_file in video_files:
        frames = extract_frames_from_video(video_file, temp_frames_dir, args.frame_interval)
        all_frames.extend(frames)
    
    if not all_frames:
        print("No frames extracted")
        return
    
    print(f"Extracted {len(all_frames)} frames total")
    
    # Run YOLO detection
    detection_results = run_yolo_on_frames(all_frames, model, args.confidence_threshold)
    
    # Organize for review
    detected_count, no_detection_count = organize_frames_for_review(detection_results, output_dir)
    
    # Create review instructions
    create_review_instructions(output_dir, detected_count, no_detection_count)
    
    # Clean up temp frames
    shutil.rmtree(temp_frames_dir)
    
    print(f"\nReview setup complete!")
    print(f"Check {output_dir}/REVIEW_INSTRUCTIONS.md for next steps")
    print(f"After human review, use animal/ and no_object/ folders for retraining")


if __name__ == "__main__":
    main()