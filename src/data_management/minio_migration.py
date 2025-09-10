#!/usr/bin/env python3
"""
Migrate MinIO bucket structure from deep nesting to Option B format.

From: raw-video/2025/09/09/16/11/47/east_fence/video-2025-09-09T16-11-47.552275.mp4
To:   2025/09/east_fence/video_20250909_161147.mp4
"""

import re
from minio import Minio
from minio.error import S3Error

def connect_to_minio():
    """Connect to MinIO server"""
    return Minio(
        '192.168.1.183:9000',
        access_key='coop',
        secret_key='coop4856',
        secure=False
    )

def parse_old_path(object_name):
    """
    Parse old path format to extract components.
    
    Input: raw-video/2025/09/09/16/11/47/east_fence/video-2025-09-09T16-11-47.552275.mp4
    Returns: (year, month, day, hour, minute, second, camera, original_filename)
    """
    # Pattern: raw-video/YYYY/MM/DD/HH/MM/SS/CAMERA/filename
    pattern = r'raw-video/(\d{4})/(\d{2})/(\d{2})/(\d{2})/(\d{2})/(\d{2})/([^/]+)/(.+)'
    match = re.match(pattern, object_name)
    
    if not match:
        return None
        
    year, month, day, hour, minute, second, camera, filename = match.groups()
    return {
        'year': year,
        'month': month, 
        'day': day,
        'hour': hour,
        'minute': minute,
        'second': second,
        'camera': camera,
        'original_filename': filename
    }

def generate_new_path(components):
    """
    Generate new path format, preserving file type.
    
    Output: 2025/09/east_fence/video_20250909_161147.mp4
    Output: 2025/09/east_fence/detections_20250909_161147.json
    """
    year = components['year']
    month = components['month']
    day = components['day']
    hour = components['hour']
    minute = components['minute']
    second = components['second']
    camera = components['camera']
    original_filename = components['original_filename']
    
    # Determine if this is a video or metadata file
    if original_filename.endswith('.json'):
        # Metadata file
        new_filename = f"detections_{year}{month}{day}_{hour}{minute}{second}.json"
    else:
        # Video file
        new_filename = f"video_{year}{month}{day}_{hour}{minute}{second}.mp4"
    
    # New path structure: YYYY/MM/camera/filename
    new_path = f"{year}/{month}/{camera}/{new_filename}"
    
    return new_path

def migrate_object(client, bucket_name, old_path, new_path, dry_run=True):
    """Migrate a single object from old to new path"""
    
    if dry_run:
        print(f"DRY RUN: Would move:")
        print(f"  FROM: {old_path}")
        print(f"  TO:   {new_path}")
        return True
    
    try:
        # Use get_object and put_object instead of copy_object to avoid CopySource issues
        
        # Download the object data
        response = client.get_object(bucket_name, old_path)
        object_data = response.data
        
        # Get object metadata
        stat = client.stat_object(bucket_name, old_path)
        content_type = stat.content_type
        
        # Create a BytesIO object for put_object
        from io import BytesIO
        data_stream = BytesIO(object_data)
        
        # Upload to new location
        client.put_object(
            bucket_name, 
            new_path, 
            data=data_stream,
            length=len(object_data),
            content_type=content_type
        )
        print(f"✅ Copied: {old_path} → {new_path}")
        
        # Remove old object
        client.remove_object(bucket_name, old_path)
        print(f"🗑️  Removed: {old_path}")
        
        return True
        
    except S3Error as e:
        print(f"❌ Error migrating {old_path}: {e}")
        return False

def main():
    print("🔄 MinIO Bucket Structure Migration")
    print("=" * 50)
    
    # Connect to MinIO
    client = connect_to_minio()
    bucket_name = 'recordings-dev'
    
    # List all objects in old format
    print(f"📋 Scanning bucket '{bucket_name}' for objects to migrate...")
    
    objects_to_migrate = []
    
    try:
        objects = client.list_objects(bucket_name, prefix='raw-video/', recursive=True)
        
        for obj in objects:
            old_path = obj.object_name
            components = parse_old_path(old_path)
            
            if components:
                new_path = generate_new_path(components)
                objects_to_migrate.append((old_path, new_path))
            else:
                print(f"⚠️  Couldn't parse: {old_path}")
    
    except S3Error as e:
        print(f"❌ Error listing objects: {e}")
        return
    
    print(f"\n📊 Found {len(objects_to_migrate)} objects to migrate")
    
    if not objects_to_migrate:
        print("✅ No objects to migrate!")
        return
    
    # Show preview of migrations
    print(f"\n👀 Preview of migrations:")
    for i, (old_path, new_path) in enumerate(objects_to_migrate[:5]):
        print(f"  {i+1}. {old_path}")
        print(f"     → {new_path}")
    
    if len(objects_to_migrate) > 5:
        print(f"     ... and {len(objects_to_migrate) - 5} more")
    
    # Ask for confirmation
    print(f"\n🤔 Ready to migrate {len(objects_to_migrate)} objects?")
    print("   This will copy objects to new structure and delete the old ones.")
    
    # First run as dry-run
    response = input("\n1. Run DRY RUN first? (y/n): ").lower().strip()
    
    if response == 'y':
        print(f"\n🧪 DRY RUN - No actual changes will be made:")
        print("-" * 40)
        
        for old_path, new_path in objects_to_migrate:
            migrate_object(client, bucket_name, old_path, new_path, dry_run=True)
        
        print(f"\n✅ Dry run complete. Ready for actual migration?")
        response = input("2. Proceed with REAL migration? (y/n): ").lower().strip()
    
    if response == 'y':
        print(f"\n🚀 Starting migration...")
        print("-" * 40)
        
        success_count = 0
        error_count = 0
        
        for old_path, new_path in objects_to_migrate:
            if migrate_object(client, bucket_name, old_path, new_path, dry_run=False):
                success_count += 1
            else:
                error_count += 1
        
        print(f"\n📊 Migration Summary:")
        print(f"  ✅ Successfully migrated: {success_count}")
        print(f"  ❌ Errors: {error_count}")
        
        if success_count > 0:
            print(f"\n🎉 Migration complete! New structure:")
            print(f"   YYYY/MM/camera/video_YYYYMMDD_HHMMSS.mp4")
            
            # Show new structure
            print(f"\n📁 Checking new structure:")
            try:
                new_objects = list(client.list_objects(bucket_name, recursive=True))
                cameras = set()
                months = set()
                
                for obj in new_objects:
                    if not obj.object_name.startswith('raw-video/'):
                        parts = obj.object_name.split('/')
                        if len(parts) >= 3:
                            months.add(f"{parts[0]}/{parts[1]}")
                            cameras.add(parts[2])
                
                print(f"   📅 Months: {sorted(months)}")
                print(f"   📹 Cameras: {sorted(cameras)}")
                
            except Exception as e:
                print(f"   ⚠️ Error checking new structure: {e}")
    
    else:
        print("❌ Migration cancelled.")

if __name__ == "__main__":
    main()