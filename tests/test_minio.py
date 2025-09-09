#!/usr/bin/env python3
"""
Simple test script to verify MinIO integration works.
Tests connection, bucket creation, file upload/download.
"""

import sys
import tempfile
from pathlib import Path
from minio import Minio
from minio.error import S3Error
import yaml

def load_config(config_path="config/cameras.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        sys.exit(1)

def test_minio_connection(config):
    """Test MinIO connection and basic operations"""
    
    # Get MinIO config
    minio_config = config.get('minio', {})
    endpoint = minio_config.get('endpoint', 'localhost:9000')
    access_key = minio_config.get('access_key', 'minioadmin')
    secret_key = minio_config.get('secret_key', 'minioadmin')
    secure = minio_config.get('secure', False)
    
    print(f"🔗 Connecting to MinIO at {endpoint}...")
    
    try:
        # Initialize MinIO client
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        
        # Test connection by listing buckets
        buckets = list(client.list_buckets())
        print(f"✅ Connected! Found {len(buckets)} existing buckets:")
        for bucket in buckets:
            print(f"   - {bucket.name}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None
        
    return client

def test_bucket_operations(client):
    """Test bucket creation and listing"""
    
    test_buckets = ['test-recordings', 'test-images']
    
    print(f"\n📦 Testing bucket operations...")
    
    for bucket_name in test_buckets:
        try:
            # Check if bucket exists
            if client.bucket_exists(bucket_name):
                print(f"✅ Bucket '{bucket_name}' already exists")
            else:
                # Create bucket
                client.make_bucket(bucket_name)
                print(f"✅ Created bucket '{bucket_name}'")
                
        except S3Error as e:
            print(f"❌ Bucket operation failed for '{bucket_name}': {e}")
            return False
            
    return True

def test_file_operations(client):
    """Test file upload/download"""
    
    print(f"\n📁 Testing file operations...")
    
    # Create a test file
    test_content = f"MinIO test file created at {Path.cwd()}\nThis is a test!"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = Path(f.name)
    
    try:
        bucket_name = 'test-recordings'
        object_name = 'test-file.txt'
        
        # Upload file
        print(f"⬆️  Uploading test file to {bucket_name}/{object_name}...")
        client.fput_object(bucket_name, object_name, str(temp_file))
        print(f"✅ Upload successful!")
        
        # List objects in bucket
        objects = list(client.list_objects(bucket_name))
        print(f"📋 Objects in {bucket_name}:")
        for obj in objects:
            print(f"   - {obj.object_name} ({obj.size} bytes)")
            
        # Download file
        download_path = temp_file.parent / "downloaded_test.txt"
        print(f"⬇️  Downloading to {download_path}...")
        client.fget_object(bucket_name, object_name, str(download_path))
        
        # Verify content
        with open(download_path, 'r') as f:
            downloaded_content = f.read()
            
        if downloaded_content == test_content:
            print(f"✅ Download successful! Content matches.")
        else:
            print(f"❌ Content mismatch!")
            return False
            
        # Cleanup
        temp_file.unlink()
        download_path.unlink()
        
        # Remove test object
        client.remove_object(bucket_name, object_name)
        print(f"🗑️  Cleaned up test file")
        
    except Exception as e:
        print(f"❌ File operation failed: {e}")
        return False
        
    return True

def main():
    """Run all tests"""
    
    print("🧪 MinIO Integration Test")
    print("=" * 40)
    
    # Load config
    config = load_config()
    
    # Test connection
    client = test_minio_connection(config)
    if not client:
        sys.exit(1)
        
    # Test bucket operations
    if not test_bucket_operations(client):
        sys.exit(1)
        
    # Test file operations  
    if not test_file_operations(client):
        sys.exit(1)
        
    print(f"\n🎉 All tests passed! MinIO integration is working correctly.")
    print(f"💡 You can now safely update your main application to use MinIO.")

if __name__ == "__main__":
    main()