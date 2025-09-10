#!/usr/bin/env python3
"""
Recover JSON metadata files to Minio after migration bug.
Uploads local recordings/*.json files to correct Minio paths.
FIXED: Handles underscores in camera names like 'east_fence'
"""

import glob
import re
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from io import BytesIO

def connect_to_minio():
    """Connect to MinIO server"""
    return Minio(
        '192.168.1.183:9000',
        access_key='coop',
        secret_key='coop4856',
        secure=False
    )

def parse_local_filename(filename):
    """
    Parse local filename to extract components.
    
    Input: recordings/east_fence_20250910_151205_detections.json
    Returns: camera, year, month, day, hour, minute, second
    """
    basename = Path(filename).name
    
    # Pattern: {camera}_{YYYYMMDD}_{HHMMSS}_detections.json
    # Use non-greedy match (.+?) to capture camera name with underscores
    pattern = r'(.+?)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_detections\.json'
    match = re.match(pattern, basename)
    
    if not match:
        return None
        
    camera, year, month, day, hour, minute, second = match.groups()
    return {
        'camera': camera,
        'year': year,
        'month': month,
        'day': day,
        'hour': hour,
        'minute': minute,
        'second': second
    }

def generate_minio_path(components):
    """Generate correct Minio path for JSON file"""
    year = components['year']
    month = components['month']
    camera = components['camera']
    day = components['day']
    hour = components['hour']
    minute = components['minute']
    second = components['second']
    
    # New path: YYYY/MM/camera/detections_YYYYMMDD_HHMMSS.json
    filename = f"detections_{year}{month}{day}_{hour}{minute}{second}.json"
    path = f"{year}/{month}/{camera}/{filename}"
    
    return path

def upload_json_file(client, bucket_name, local_path, minio_path):
    """Upload JSON file to Minio"""
    
    try:
        # Read local file
        with open(local_path, 'rb') as f:
            json_data = f.read()
        
        # Create BytesIO stream
        json_stream = BytesIO(json_data)
        
        # Upload to Minio
        client.put_object(
            bucket_name=bucket_name,
            object_name=minio_path,
            data=json_stream,
            length=len(json_data),
            content_type='application/json'
        )
        
        print(f"✅ Uploaded: {local_path} → {minio_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading {local_path}: {e}")
        return False

def main():
    print("🔧 JSON Metadata Recovery Tool (FIXED)")
    print("=" * 50)
    
    # Connect to Minio
    print("Connecting to Minio...")
    try:
        client = connect_to_minio()
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    bucket_name = 'recordings-dev'
    
    # Find all local JSON files
    json_files = glob.glob('recordings/*_detections.json')
    
    if not json_files:
        print("❌ No JSON files found in recordings/ directory")
        return
    
    print(f"📋 Found {len(json_files)} JSON files to recover")
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for json_file in json_files:
        print(f"\n📄 Processing: {json_file}")
        
        # Parse filename
        components = parse_local_filename(json_file)
        if not components:
            print(f"⚠️  Could not parse filename: {json_file}")
            error_count += 1
            continue
        
        # Generate Minio path
        minio_path = generate_minio_path(components)
        print(f"   Target: {minio_path}")
        
        # Upload file
        if upload_json_file(client, bucket_name, json_file, minio_path):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\n📊 Recovery Summary:")
    print(f"  ✅ Successfully uploaded: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    
    if success_count > 0:
        print(f"\n🎉 JSON metadata recovery complete!")
        print(f"   Check Minio web portal - you should now see .json files alongside .mp4 files")

if __name__ == "__main__":
    main()