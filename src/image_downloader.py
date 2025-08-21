# v43 - Streamlined direct download approach
import os
import json
import logging
from typing import Dict, List, Tuple
from geti_client import GetiClient

logger = logging.getLogger(__name__)


class GetiImageDownloader:
    """Download images from Intel's Geti platform using direct approach"""
    
    def __init__(self, host: str, project_name: str):
        """Initialize downloader with Geti connection"""
        self.geti_client = GetiClient(host=host, project_name=project_name)
        logger.info(f"Connected to Geti project: {project_name}")
    
    def quick_test_download(self, output_images_dir: str, num_images: int = 20):
        """Quick test download of a few images without processing full dataset"""
        os.makedirs(output_images_dir, exist_ok=True)
        
        logger.info(f"🧪 QUICK TEST: Downloading {num_images} images for testing...")
        
        # Get just the images we need (much faster)
        logger.info(f"Getting first {num_images} images...")
        test_images = self.geti_client.get_sample_images(max_images=num_images)
        
        downloaded_count = 0
        error_count = 0
        
        for i, geti_image in enumerate(test_images):
            logger.info(f"Quick test progress: {i+1}/{len(test_images)} - {geti_image.name}")
            
            try:
                # Use sanitized filename
                safe_filename = self._sanitize_filename(geti_image.name) + ".jpg"
                output_path = os.path.join(output_images_dir, safe_filename)
                
                # Use the working download method with timeout
                download_url = geti_image.download_url
                if download_url.startswith('/'):
                    download_url = "https://app.geti.intel.com" + download_url
                
                import requests
                response = requests.get(download_url, headers=self.geti_client.geti.session.headers, timeout=30)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                file_size = len(response.content)
                logger.info(f"✅ Downloaded {safe_filename} ({file_size:,} bytes)")
                downloaded_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error downloading {geti_image.name}: {e}")
                error_count += 1
        
        logger.info(f"🧪 Quick test complete!")
        logger.info(f"   Downloaded: {downloaded_count}")
        logger.info(f"   Errors: {error_count}")
        logger.info(f"   Output: {output_images_dir}")
        
        return downloaded_count, error_count
    
    def download_coco_images_direct(self, output_images_dir: str, test_mode: bool = False, test_limit: int = 50):
        """
        Download images directly from current annotations (streamlined v40 approach)
        """
        os.makedirs(output_images_dir, exist_ok=True)
        
        # Get images from Geti for mapping
        logger.info("Fetching image list from Geti...")
        geti_images = self.geti_client.get_all_images()
        geti_image_map = {img.name: img for img in geti_images}
        
        # Extract annotations directly to get current image names  
        logger.info("Extracting annotations to get current image names...")
        if test_mode:
            logger.info(f"🧪 TEST MODE: Only processing first {test_limit} images for speed")
            annotations = self.geti_client.extract_annotations(confidence_threshold=0.7, max_images=test_limit)
        else:
            logger.info("⚡ Processing all annotations (this may take ~20 minutes)...")
            annotations = self.geti_client.extract_annotations(confidence_threshold=0.7)
        
        # Get unique images from current annotations (primary approach - no ID mapping!)
        unique_images = {}
        for ann in annotations:
            if ann.image_name not in unique_images:
                unique_images[ann.image_name] = geti_image_map.get(ann.image_name)
        
        logger.info(f"Found {len(unique_images)} unique images from current annotations")
        
        # Limit for testing if needed
        if test_mode:
            items = list(unique_images.items())[:test_limit]
            logger.info(f"🧪 TEST MODE: Downloading first {test_limit} images out of {len(unique_images)} available")
        else:
            items = list(unique_images.items())
        
        downloaded_count = 0
        error_count = 0
        
        for i, (image_name, geti_image) in enumerate(items):
            if i % 10 == 0 if test_mode else i % 50 == 0:
                logger.info(f"v43 Download progress: {i}/{len(items)} images...")
            
            if geti_image is not None:
                try:
                    # Use working method with timeout and retry
                    safe_filename = self._sanitize_filename(image_name) + ".jpg"
                    output_path = os.path.join(output_images_dir, safe_filename)
                    
                    # Retry logic for network issues
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            download_url = geti_image.download_url
                            if download_url.startswith('/'):
                                download_url = "https://app.geti.intel.com" + download_url
                            
                            import requests
                            response = requests.get(
                                download_url, 
                                headers=self.geti_client.geti.session.headers,
                                timeout=30
                            )
                            response.raise_for_status()
                            
                            with open(output_path, 'wb') as f:
                                f.write(response.content)
                            
                            downloaded_count += 1
                            break  # Success - exit retry loop
                            
                        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {image_name} - retrying...")
                                import time
                                time.sleep(2)
                                continue
                            else:
                                raise e
                        
                except Exception as e:
                    logger.error(f"Error downloading {image_name}: {e}")
                    error_count += 1
            else:
                logger.warning(f"Geti image not found: {image_name}")
                error_count += 1
        
        logger.info(f"✅ v42 Download complete!")
        logger.info(f"   Downloaded: {downloaded_count}")
        logger.info(f"   Errors: {error_count}")
        logger.info(f"   Output: {output_images_dir}")
        
        return downloaded_count, error_count
    
    def _sanitize_filename(self, filename: str) -> str:
        """Convert Geti image name to filesystem-safe filename while preserving semantic info"""
        # Extract just the meaningful part (remove path prefixes)
        if '/' in filename:
            filename = filename.split('/')[-1]
        
        # Replace filesystem-unsafe characters
        safe_name = filename.replace('/', '_').replace('\\', '_').replace(':', '-').replace('.', '-')
        
        # Remove any double underscores or dashes
        while '__' in safe_name:
            safe_name = safe_name.replace('__', '_')
        while '--' in safe_name:
            safe_name = safe_name.replace('--', '-')
        
        # Ensure it doesn't start/end with special chars
        safe_name = safe_name.strip('_-')
        
        return safe_name


def quick_test_download(host: str, project_name: str, num_images: int = 20):
    """Super fast test - just download first N images without any COCO processing"""
    downloader = GetiImageDownloader(host=host, project_name=project_name)
    return downloader.quick_test_download("test_images", num_images)


def download_images_direct(host: str, project_name: str, output_dir: str, test_mode: bool = False, test_limit: int = 50):
    """
    Direct download approach (v40) - no COCO ID mapping complexity
    """
    downloader = GetiImageDownloader(host=host, project_name=project_name)
    images_dir = os.path.join(output_dir, 'images')
    
    if test_mode:
        logger.info(f"🧪 TEST MODE: Will download first {test_limit} images")
    
    return downloader.download_coco_images_direct(images_dir, test_mode, test_limit)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # v43 STREAMLINED APPROACH
    print("🚀 Starting v43 direct download...")
    downloaded, errors = download_images_direct(
        host="https://app.geti.intel.com",
        project_name="Wildlife #3",
        output_dir="wildlife_coco_dataset",
        test_mode=False,  # Set to True for quick test
    #     test_mode=True,  # Set to True for quick test
    #     test_limit=2500
    )
    
    print(f"\n✅ v43 Download complete!")
    print(f"📥 Downloaded: {downloaded} images")
    print(f"❌ Errors: {errors} images")
    print(f"📁 Location: wildlife_coco_dataset/images/")