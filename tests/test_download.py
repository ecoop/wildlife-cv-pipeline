import os
import sys
import logging

# Add src directory to path so we can import geti_client
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from geti_client import GetiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_image_download():
    """Test downloading a single image to debug the API"""
    
    # Connect to Geti
    client = GetiClient(
        host="https://app.geti.intel.com",
        project_name="Wildlife #3"
    )
    
    # Get first few images
    logger.info("Getting first few images...")
    all_images = client.get_all_images()
    test_images = all_images[:3]  # Just test first 3
    
    for i, image in enumerate(test_images):
        logger.info(f"\n=== TESTING IMAGE {i+1}: {image.name} ===")
        logger.info(f"Image ID: {image.id}")
        logger.info(f"Image type: {type(image)}")
        
        # Show available attributes
        logger.info(f"Available attributes: {[attr for attr in dir(image) if not attr.startswith('_')]}")
        
        # Test each download method
        test_filename = f"test_image_{i+1}.jpg"
        
        # Method 1: get_data with session
        try:
            logger.info("Testing Method 1: image.get_data(session)")
            image_data = image.get_data(session=client.geti.session)
            logger.info(f"get_data returned: {type(image_data)}, length: {len(image_data) if hasattr(image_data, '__len__') else 'unknown'}")
            
            if isinstance(image_data, bytes):
                with open(test_filename, 'wb') as f:
                    f.write(image_data)
                file_size = os.path.getsize(test_filename)
                logger.info(f"✅ SUCCESS! Saved {test_filename} ({file_size} bytes)")
                return True
            else:
                logger.info(f"❌ get_data returned {type(image_data)}, not bytes")
                
        except Exception as e:
            logger.info(f"❌ Method 1 failed: {e}")
        
        # Method 2: Try download_url
        try:
            if hasattr(image, 'download_url') and image.download_url:
                logger.info(f"Testing Method 2: download_url = {image.download_url}")
                
                # Fix URL if needed
                download_url = image.download_url
                if download_url.startswith('/'):
                    download_url = "https://app.geti.intel.com" + download_url
                
                import requests
                response = requests.get(download_url, headers=client.geti.session.headers)
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response content-type: {response.headers.get('content-type')}")
                logger.info(f"Response length: {len(response.content)}")
                
                if response.status_code == 200:
                    with open(test_filename, 'wb') as f:
                        f.write(response.content)
                    file_size = os.path.getsize(test_filename)
                    logger.info(f"✅ SUCCESS! Saved {test_filename} ({file_size} bytes)")
                    return True
                    
        except Exception as e:
            logger.info(f"❌ Method 2 failed: {e}")
        
        # Method 3: Try base_url  
        try:
            if hasattr(image, 'base_url') and image.base_url:
                logger.info(f"Testing Method 3: base_url = {image.base_url}")
                
                # Fix URL if needed
                base_url = image.base_url
                if base_url.startswith('/'):
                    base_url = "https://app.geti.intel.com" + base_url
                
                import requests
                response = requests.get(base_url, headers=client.geti.session.headers)
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response content-type: {response.headers.get('content-type')}")
                logger.info(f"Response length: {len(response.content)}")
                
                if response.status_code == 200:
                    with open(test_filename, 'wb') as f:
                        f.write(response.content)
                    file_size = os.path.getsize(test_filename)
                    logger.info(f"✅ SUCCESS! Saved {test_filename} ({file_size} bytes)")
                    return True
                    
        except Exception as e:
            logger.info(f"❌ Method 3 failed: {e}")
        
        # Method 4: Try image client methods
        try:
            logger.info("Testing Method 4: image_client methods")
            image_client = client.image_client
            
            # Try different client methods
            for method_name in ['download_image', 'get_image_data', 'download']:
                if hasattr(image_client, method_name):
                    try:
                        logger.info(f"Trying image_client.{method_name}")
                        method = getattr(image_client, method_name)
                        result = method(image)
                        
                        logger.info(f"{method_name} returned: {type(result)}")
                        
                        if isinstance(result, bytes):
                            with open(test_filename, 'wb') as f:
                                f.write(result)
                            file_size = os.path.getsize(test_filename)
                            logger.info(f"✅ SUCCESS! Saved {test_filename} ({file_size} bytes)")
                            return True
                            
                    except Exception as e:
                        logger.info(f"❌ {method_name} failed: {e}")
                        
        except Exception as e:
            logger.info(f"❌ Method 4 failed: {e}")
        
        # Method 5: Try numpy conversion
        try:
            if hasattr(image, 'numpy'):
                logger.info("Testing Method 5: numpy conversion")
                img_array = image.numpy
                logger.info(f"numpy returned: {type(img_array)}")
                
                if img_array is not None:
                    from PIL import Image as PILImage
                    pil_image = PILImage.fromarray(img_array)
                    pil_image.save(test_filename, 'JPEG', quality=90)
                    file_size = os.path.getsize(test_filename)
                    logger.info(f"✅ SUCCESS! Saved {test_filename} ({file_size} bytes)")
                    return True
                    
        except Exception as e:
            logger.info(f"❌ Method 5 failed: {e}")
        
        logger.info(f"❌ All methods failed for {image.name}")
    
    logger.info("\n❌ No download method worked for any test image")
    return False


if __name__ == "__main__":
    success = test_single_image_download()
    if success:
        print("\n🎉 Found working download method!")
    else:
        print("\n😞 No download method worked. Need to investigate Geti SDK documentation.")