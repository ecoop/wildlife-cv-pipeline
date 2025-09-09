#!/usr/bin/env python3
"""
Minio Ingestion Tool for Wildlife CV Pipeline

Ingests existing datasets into Minio with unified metadata schema.
Extracts metadata from legacy filenames and COCO annotations.

Usage:
    python src/data_management/minio_ingestion.py --dataset_path ~/datasets/wildlife_coco_dataset --bucket wildlife-data
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib
from minio import Minio
from minio.error import S3Error
import cv2
from tqdm import tqdm


def parse_wildlife_filename(filename: str) -> Optional[Dict]:
    """
    Parse metadata from wildlife dataset filenames
    
    Pattern: {camera_id}_{YYYY}_{MM}_{DD}_{HH}-{MM}-{SS}_{?}_mp4-{frame}_jpg-rf-{hash}.jpg
    Example: 249_2023_06_29_11-36-45_3_mp4-497_jpg-rf-acb873c3fedd2fd9e8b99053247fd0b5.jpg
    
    Camera ID mapping:
    - 249 = FF0A 
    - 250 = D161
    """
    
    # Camera ID mapping from numeric to hex identifiers
    CAMERA_ID_MAPPING = {
        '249': 'FF0A',
        '250': 'D161'
    }
    
    pattern = r'(\d+)_(\d{4})_(\d{2})_(\d{2})_(\d{2})-(\d{2})-(\d{2})_(\d+)_mp4-(\d+)_jpg-rf-([a-f0-9]+)\.jpg'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    numeric_camera_id, year, month, day, hour, minute, second, unknown, frame, hash_val = match.groups()
    
    # Map numeric camera ID to hex identifier
    camera_id = CAMERA_ID_MAPPING.get(numeric_camera_id, f"cam{numeric_camera_id}")
    
    timestamp = f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"
    source_video = f"{numeric_camera_id}_{year}_{month}_{day}_{hour}-{minute}-{second}_{unknown}.mp4"
    
    return {
        'camera_id': camera_id,
        'numeric_camera_id': numeric_camera_id,
        'timestamp': timestamp,
        'frame_number': int(frame),
        'source_video': source_video,
        'original_hash': hash_val
    }


def get_image_metadata(image_path: Path) -> Dict:
    """Extract basic image metadata"""
    
    img = cv2.imread(str(image_path))
    if img is None:
        return {}
    
    height, width = img.shape[:2]
    file_size = image_path.stat().st_size
    
    # Calculate file checksum
    with open(image_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    return {
        'width': width,
        'height': height, 
        'file_size_bytes': file_size,
        'checksum': f"sha256:{file_hash}"
    }


def convert_coco_to_unified_schema(image_info: Dict, annotations: List[Dict], categories: Dict) -> Dict:
    """Convert COCO format to unified metadata schema"""
    
    # Parse filename metadata
    filename_meta = parse_wildlife_filename(image_info['file_name'])
    if not filename_meta:
        print(f"Warning: Could not parse filename {image_info['file_name']}")
        filename_meta = {}
    
    # Convert COCO annotations
    detections = []
    for ann in annotations:
        category = categories.get(ann['category_id'], {})
        
        detection = {
            'id': ann['id'],
            'category': 'animal',  # Generic category
            'species': category.get('name', 'unknown'),
            'confidence': 1.0,  # COCO annotations are ground truth
            'bbox': ann['bbox'],  # [x, y, width, height]
            'bbox_format': 'xywh',
            'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
            'created_at': datetime.now().isoformat() + 'Z',
            'created_by': 'human_annotation_geti'
        }
        detections.append(detection)
    
    # Build unified schema
    unified_metadata = {
        'schema_version': '1.0',
        'media_info': {
            'media_type': 'image',
            'format': 'jpg',
            'width': image_info['width'],
            'height': image_info['height'],
            'file_size_bytes': None,  # Will be filled when image is processed
            'checksum': None  # Will be filled when image is processed
        },
        'capture_info': {
            'camera_id': filename_meta.get('camera_id', 'unknown'),
            'camera_location': {
                'name': f"camera_{filename_meta.get('camera_id', 'unknown')}",
                'coordinates': None  # Could be populated from camera config
            },
            'timestamp': filename_meta.get('timestamp'),
            'source_video': filename_meta.get('source_video'),
            'frame_number': filename_meta.get('frame_number'),
            'extraction_method': 'geti_export',
            'original_hash': filename_meta.get('original_hash')
        },
        'annotations': {
            'format': 'coco_extended',
            'annotator': 'human',
            'confidence_threshold': None,
            'review_status': 'approved',  # COCO data is pre-reviewed
            'detections': detections
        },
        'processing_history': [
            {
                'step': 'geti_export',
                'timestamp': datetime.now().isoformat() + 'Z',
                'tool': 'geti_to_coco_exporter',
                'version': '1.0'
            }
        ],
        'project_info': {
            'project': 'wildlife-cv-pipeline',
            'subproject': 'initial-dataset',
            'dataset_version': 'v1.0',
            'split': 'train'  # Default, could be determined later
        }
    }
    
    return unified_metadata


def create_minio_path(metadata: Dict, image_filename: str) -> str:
    """Create Minio object path following our structure"""
    
    timestamp = metadata['capture_info'].get('timestamp')
    camera_id = metadata['capture_info'].get('camera_id', 'unknown')
    
    if timestamp:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        path = f"processed-images/project-wildlife-cv/dataset-v1.0/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{dt.hour:02d}/{dt.minute:02d}/{dt.second:02d}/{camera_id}/{image_filename}"
    else:
        path = f"processed-images/project-wildlife-cv/dataset-v1.0/unknown-date/{camera_id}/{image_filename}"
    
    return path


def setup_minio_client(endpoint: str = "localhost:9000", access_key: str = "minioadmin", secret_key: str = "minioadmin") -> Minio:
    """Setup Minio client with default local settings"""
    
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False  # Set to True for HTTPS
    )
    
    return client


def ingest_coco_dataset(dataset_path: Path, minio_client: Minio, bucket_name: str):
    """Ingest COCO dataset into Minio with unified schema"""
    
    annotations_path = dataset_path / 'annotations.json'
    images_dir = dataset_path / 'images'
    
    # Load COCO data
    print("Loading COCO annotations...")
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build lookup tables
    categories = {cat['id']: cat for cat in coco_data['categories']}
    images_lookup = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Ensure bucket exists
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Created bucket: {bucket_name}")
    except S3Error as e:
        print(f"Error creating bucket: {e}")
        return
    
    # Process each image
    processed = 0
    skipped = 0
    
    print(f"Processing {len(coco_data['images'])} images...")
    
    for image_info in tqdm(coco_data['images']):
        image_id = image_info['id']
        filename = image_info['file_name']
        image_path = images_dir / filename
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            skipped += 1
            continue
        
        # Get annotations for this image
        image_annotations = annotations_by_image.get(image_id, [])
        
        # Convert to unified schema
        unified_metadata = convert_coco_to_unified_schema(image_info, image_annotations, categories)
        
        # Get actual image metadata
        img_meta = get_image_metadata(image_path)
        unified_metadata['media_info'].update(img_meta)
        
        # Create Minio paths
        image_object_path = create_minio_path(unified_metadata, filename)
        metadata_object_path = image_object_path.replace('.jpg', '.json')
        
        try:
            # Upload image
            minio_client.fput_object(
                bucket_name,
                image_object_path,
                str(image_path),
                content_type='image/jpeg'
            )
            
            # Upload metadata
            metadata_json = json.dumps(unified_metadata, indent=2)
            minio_client.put_object(
                bucket_name,
                metadata_object_path,
                data=metadata_json.encode('utf-8'),
                length=len(metadata_json.encode('utf-8')),
                content_type='application/json'
            )
            
            processed += 1
            
        except S3Error as e:
            print(f"Error uploading {filename}: {e}")
            skipped += 1
    
    print(f"\nIngestion complete:")
    print(f"  Processed: {processed} images")
    print(f"  Skipped: {skipped} images")
    print(f"  Bucket: {bucket_name}")


def main():
    parser = argparse.ArgumentParser(description='Ingest datasets into Minio')
    parser.add_argument('--dataset_path', required=True, help='Path to COCO dataset directory')
    parser.add_argument('--bucket', default='wildlife-data', help='Minio bucket name')
    parser.add_argument('--minio_endpoint', default='localhost:9000', help='Minio endpoint')
    parser.add_argument('--access_key', default='minioadmin', help='Minio access key')
    parser.add_argument('--secret_key', default='minioadmin', help='Minio secret key')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path).expanduser()
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        return
    
    # Setup Minio client
    print("Connecting to Minio...")
    try:
        minio_client = setup_minio_client(args.minio_endpoint, args.access_key, args.secret_key)
        print("Connected successfully")
    except Exception as e:
        print(f"Error connecting to Minio: {e}")
        return
    
    # Ingest dataset
    ingest_coco_dataset(dataset_path, minio_client, args.bucket)


if __name__ == "__main__":
    main()