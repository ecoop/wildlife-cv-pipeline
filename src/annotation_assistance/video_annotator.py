#!/usr/bin/env python3
"""
Video annotation system for wildlife detection recordings.

Downloads raw videos and JSON metadata from Minio, applies bounding box overlays,
and uploads annotated videos to the annotated-video prefix.

Usage:
    python src/annotation/video_annotator.py --camera east_fence --date 2025-09-11
    python src/annotation/video_annotator.py --camera west_fence --date-range 2025-09-01:2025-09-11
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error
import re
from typing import List, Dict, Optional, Tuple
import yaml

class MinioVideoAnnotator:
    def __init__(self, config_path: str = "config/cameras.yaml"):
        """Initialize with Minio connection from config"""
        self.logger = logging.getLogger(__name__)
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        minio_config = config.get('minio', {})
        
        # Connect to Minio
        self.client = Minio(
            minio_config.get('endpoint', 'localhost:9000'),
            access_key=minio_config.get('access_key', 'minioadmin'),
            secret_key=minio_config.get('secret_key', 'minioadmin'),
            secure=minio_config.get('secure', False)
        )
        
        self.bucket = f"recordings-{minio_config.get('environment', 'dev')}"
        self.logger.info(f"Connected to Minio bucket: {self.bucket}")

    def search_recordings(self, camera: Optional[str] = None, 
                         date: Optional[str] = None,
                         date_range: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Search for video/JSON pairs in raw-video prefix.
        
        Args:
            camera: Camera name to filter (e.g., 'east_fence')
            date: Specific date YYYY-MM-DD
            date_range: Date range YYYY-MM-DD:YYYY-MM-DD
            
        Returns:
            List of {video_path, json_path, camera, timestamp} dicts
        """
        recordings = []
        
        try:
            # List all objects in raw-video prefix
            objects = self.client.list_objects(self.bucket, prefix='raw-video/', recursive=True)
            
            # Group by video/json pairs
            video_files = {}
            json_files = {}
            
            for obj in objects:
                path = obj.object_name
                
                # Parse path: raw-video/YYYY/MM/camera/filename
                match = re.match(r'raw-video/(\d{4})/(\d{2})/([^/]+)/(.+)', path)
                if not match:
                    continue
                    
                year, month, cam, filename = match.groups()
                
                # Apply camera filter
                if camera and cam != camera:
                    continue
                
                # Apply date filters
                file_date = f"{year}-{month}"
                if date:
                    target_date = date[:7]  # YYYY-MM
                    if file_date != target_date:
                        continue
                        
                if date_range:
                    start_date, end_date = date_range.split(':')
                    start_month = start_date[:7]
                    end_month = end_date[:7]
                    if not (start_month <= file_date <= end_month):
                        continue
                
                # Extract timestamp from filename
                timestamp_match = re.search(r'(\d{8})_(\d{6})', filename)
                if not timestamp_match:
                    continue
                    
                date_part, time_part = timestamp_match.groups()
                timestamp = f"{date_part}_{time_part}"
                key = f"{cam}_{timestamp}"
                
                if filename.startswith('video_'):
                    video_files[key] = path
                elif filename.startswith('detections_'):
                    json_files[key] = path
            
            # Match video/json pairs
            for key in video_files:
                if key in json_files:
                    # Key format: camera_YYYYMMDD_HHMMSS  
                    # Split on the date part (first 8 digits after underscore)
                    parts = key.split('_')
                    if len(parts) >= 3:
                        # Find where the date starts (8 consecutive digits)
                        for i, part in enumerate(parts):
                            if len(part) == 8 and part.isdigit():
                                cam = '_'.join(parts[:i])
                                timestamp = '_'.join(parts[i:])
                                break
                        else:
                            continue  # Couldn't find date pattern
                    else:
                        continue
                    recordings.append({
                        'video_path': video_files[key],
                        'json_path': json_files[key],
                        'camera': cam,
                        'timestamp': timestamp
                    })
            
            self.logger.info(f"Found {len(recordings)} video/JSON pairs")
            return recordings
            
        except S3Error as e:
            self.logger.error(f"Error searching recordings: {e}")
            return []

    def download_file(self, object_path: str) -> bytes:
        """Download file from Minio and return bytes"""
        try:
            response = self.client.get_object(self.bucket, object_path)
            data = response.data
            response.close()
            return data
        except S3Error as e:
            self.logger.error(f"Error downloading {object_path}: {e}")
            raise

    def parse_detections(self, json_data: bytes) -> Tuple[List[Dict], float, float]:
        """Parse JSON detection data and return detections with timing info"""
        try:
            data = json.loads(json_data.decode('utf-8'))
            
            # Extract timing information
            start_time = data.get('start_timestamp', 0.0)
            end_time = data.get('end_timestamp', 0.0)
            
            # Handle different JSON formats
            if 'detections' in data:
                return data['detections'], start_time, end_time
            elif isinstance(data, list):
                return data, start_time, end_time
            else:
                self.logger.warning("Unknown JSON format")
                return [], start_time, end_time
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}")
            return [], 0.0, 0.0

    def annotate_video(self, video_data: bytes, detections: List[Dict], 
                      start_time: float, end_time: float, output_path: str) -> bool:
        """
        Create annotated video with bounding boxes.
        
        Args:
            video_data: Raw video bytes
            detections: List of detection objects
            start_time: Video start timestamp
            end_time: Video end timestamp  
            output_path: Local path to save annotated video
            
        Returns:
            True if successful
        """
        # Save video data to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video.write(video_data)
            temp_video_path = temp_video.name

        try:
            # Open video
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                self.logger.error("Could not open video")
                return False

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Video properties: fps={fps}, width={width}, height={height}, frames={total_frames}")
            
            # Validate and fix FPS
            if fps <= 0 or fps != fps:  # Check for 0, negative, or NaN
                fps = 5.0  # Default to 5 FPS (from config)
                self.logger.warning(f"Invalid FPS detected, using default: {fps}")
            else:
                fps = max(1.0, fps)  # Ensure at least 1 FPS
            
            fps = int(fps)
            self.logger.info(f"Using FPS: {fps}")

            # Use mp4v codec which is most compatible with QuickTime
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                self.logger.error(f"Failed to open video writer for {output_path}")
                return False

            frame_idx = 0
            video_duration = end_time - start_time
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate current timestamp for this frame
                if fps > 0 and total_frames > 0:
                    # Simple linear interpolation: frame position as fraction of total duration
                    frame_progress = frame_idx / total_frames
                    current_time = start_time + (frame_progress * video_duration)
                else:
                    current_time = start_time
                
                # Find detections within a small time window of current frame (~0.5 seconds)
                time_tolerance = 0.5
                frame_detections = [d for d in detections 
                                  if abs(d.get('timestamp', 0) - current_time) <= time_tolerance]

                # Draw bounding boxes
                for detection in frame_detections:
                    bbox = detection.get('bbox', [])
                    if len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                        x, y, w, h = int(x), int(y), int(w), int(h)
                    else:
                        continue  # Skip invalid bbox
                    
                    confidence = detection.get('confidence', 0.0)
                    label = detection.get('class_name', 'unknown')
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    text = f"{label} {confidence:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                                (x + text_size[0], y), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                out.write(frame)
                frame_idx += 1

            cap.release()
            out.release()
            Path(temp_video_path).unlink()  # Clean up temp file
            
            self.logger.info(f"Annotated video saved: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error annotating video: {e}")
            Path(temp_video_path).unlink(missing_ok=True)
            return False

    def upload_annotated_video(self, local_path: str, minio_path: str, source_modified_time: Optional[datetime] = None) -> bool:
        """Upload annotated video to annotated-video prefix"""
        try:
            metadata = {}
            if source_modified_time:
                # Set custom metadata to preserve source timestamp
                metadata['X-Amz-Meta-Source-Modified'] = source_modified_time.isoformat()
                
            self.client.fput_object(
                self.bucket, 
                minio_path, 
                local_path, 
                content_type='video/mp4',
                metadata=metadata
            )
            self.logger.info(f"Uploaded: {local_path} → {minio_path}")
            return True
        except S3Error as e:
            self.logger.error(f"Error uploading {minio_path}: {e}")
            return False

    def process_recording(self, recording: Dict[str, str]) -> bool:
        """Process a single video/JSON pair"""
        video_path = recording['video_path']
        json_path = recording['json_path']
        camera = recording['camera']
        timestamp = recording['timestamp']
        
        self.logger.info(f"Processing {camera} {timestamp}")
        
        try:
            # Get source file timestamp for preservation
            try:
                source_stat = self.client.stat_object(self.bucket, video_path)
                source_modified_time = source_stat.last_modified
            except:
                source_modified_time = None
            
            # Download files
            video_data = self.download_file(video_path)
            json_data = self.download_file(json_path)
            
            # Parse detections
            detections, start_time, end_time = self.parse_detections(json_data)
            if not detections:
                self.logger.warning(f"No detections found in {json_path}")
            
            # Create annotated video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            if not self.annotate_video(video_data, detections, start_time, end_time, temp_output_path):
                return False
            
            # Generate output path: annotated-video/YYYY/MM/camera/annotated_YYYYMMDD_HHMMSS.mp4
            video_filename = Path(video_path).name
            annotated_filename = video_filename.replace('video_', 'annotated_')
            
            # Extract year/month from original path
            path_parts = video_path.split('/')
            year, month = path_parts[1], path_parts[2]
            
            output_path = f"annotated-video/{year}/{month}/{camera}/{annotated_filename}"
            
            # Upload annotated video
            success = self.upload_annotated_video(temp_output_path, output_path, source_modified_time)
            
            # Clean up
            Path(temp_output_path).unlink(missing_ok=True)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing {video_path}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Annotate wildlife detection videos')
    parser.add_argument('--config', default='config/cameras.yaml', 
                       help='YAML config file path')
    parser.add_argument('--camera', help='Camera name to filter')
    parser.add_argument('--date', help='Specific date YYYY-MM-DD')
    parser.add_argument('--date-range', help='Date range YYYY-MM-DD:YYYY-MM-DD')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create annotator
    annotator = MinioVideoAnnotator(args.config)
    
    # Search for recordings
    recordings = annotator.search_recordings(
        camera=args.camera,
        date=args.date, 
        date_range=args.date_range
    )
    
    if not recordings:
        print("No matching recordings found")
        return
    
    print(f"Found {len(recordings)} recordings to process")
    
    # Process each recording
    success_count = 0
    for recording in recordings:
        if annotator.process_recording(recording):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(recordings)} recordings")

if __name__ == "__main__":
    main()