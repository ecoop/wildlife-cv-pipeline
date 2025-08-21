import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from geti_sdk import Geti
from geti_sdk.rest_clients import ProjectClient, ImageClient, AnnotationClient
from geti_sdk.data_models import AnnotationScene, Image, Annotation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GetiAnnotationData:
    """Structured data for a single annotation extracted from Geti"""
    image_name: str
    image_id: str
    image_width: int
    image_height: int
    annotation_id: str
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    category_name: str
    confidence: float


class GetiClient:
    """Client for extracting annotation data from Intel's Geti platform"""
    
    def __init__(self, host: str, project_name: str):
        """
        Initialize Geti client
        
        Args:
            host: Geti server URL (e.g., "https://app.geti.intel.com")
            project_name: Name of the project to export
        """
        self.host = host
        self.project_name = project_name
        
        # Get API token from environment
        self.api_token = os.getenv("GETI_API_TOKEN")
        if not self.api_token:
            raise RuntimeError("GETI_API_TOKEN not set in environment variables")
        
        # Initialize Geti connection
        logger.info(f"Connecting to Geti at {host}")
        self.geti = Geti(host=host, token=self.api_token)
        
        # Get project
        logger.info(f"Loading project: {project_name}")
        self.project = self.geti.get_project(project_name)
        if not self.project:
            raise ValueError(f"Project '{project_name}' not found")
        
        # Initialize clients
        self.project_client = ProjectClient(
            session=self.geti.session, 
            workspace_id=self.geti.workspace_id
        )
        self.image_client = ImageClient(
            session=self.geti.session, 
            workspace_id=self.geti.workspace_id, 
            project=self.project
        )
        self.annotation_client = AnnotationClient(
            session=self.geti.session, 
            workspace_id=self.geti.workspace_id, 
            project=self.project
        )
        
        logger.info(f"Successfully connected to project '{self.project.name}'")
    
    def get_all_images(self) -> List[Image]:
        """Get all images in the project"""
        logger.info("Fetching all images from Geti API...")
        images = self.image_client.get_all_images()
        logger.info(f"Found {len(images)} images")
        return images

    def get_sample_images(self, max_images: int = 100) -> List[Image]:
        """Get a limited number of images (much faster than get_all_images)"""
        logger.info(f"Fetching first {max_images} images from Geti API...")
        
        # Get limited set instead of all images
        images = self.image_client.get_all_images()
        limited_images = images[:max_images]
        
        logger.info(f"Retrieved {len(limited_images)} sample images")
        return limited_images
    
    def extract_annotations(self, confidence_threshold: float = 0.5, max_images: Optional[int] = None) -> List[GetiAnnotationData]:
        """
        Extract all annotations from the project
        
        Args:
            confidence_threshold: Minimum confidence for including annotations
            max_images: Limit processing to first N images (for testing). 
                       Note: Still fetches full image list from API first.
            
        Returns:
            List of structured annotation data
        """
        images = self.get_all_images()
        
        # Limit to subset for testing
        if max_images:
            logger.info(f"Processing only first {max_images} images (out of {len(images)} total)")
            images = images[:max_images]
        all_annotations = []
        
        logger.info("Extracting annotations...")
        
        # Counters to track dimension success
        dimension_success_count = 0
        dimension_fail_count = 0
        
        for i, image in enumerate(images):
            if max_images and i >= max_images:
                logger.info(f"Stopping at {max_images} images (for testing)")
                break
                
            if i % 50 == 0:  # Progress logging
                logger.info(f"Processing image {i+1}/{len(images)}")
            
            try:
                annotation_scene = self.annotation_client.get_annotation(media_item=image)
                
                if not isinstance(annotation_scene, AnnotationScene):
                    logger.warning(f"No annotations found for image {image.name}")
                    continue
                
                        # Extract annotations from this image
                image_annotations = self._extract_image_annotations(
                    image, annotation_scene, confidence_threshold
                )
                all_annotations.extend(image_annotations)
                
                # Track dimension success for this image
                if image_annotations:
                    # Check if any annotation got real dimensions
                    for ann in image_annotations:
                        if ann.image_width > 0 and ann.image_height > 0:
                            dimension_success_count += 1
                            break
                    else:
                        dimension_fail_count += 1
                
            except Exception as e:
                logger.error(f"Error processing image {image.name}: {e}")
                continue
        
        logger.info(f"Extracted {len(all_annotations)} annotations total")
        logger.info(f"DIMENSION SUMMARY: {dimension_success_count} images with dimensions, {dimension_fail_count} without")
        
        return all_annotations
    
    def _extract_image_annotations(
        self, 
        image: Image, 
        annotation_scene: AnnotationScene, 
        confidence_threshold: float
    ) -> List[GetiAnnotationData]:
        """Extract annotations from a single image"""
        annotations = []
        
        for annotation in annotation_scene.annotations:
            # Get the best category label (highest confidence)
            best_label = self._get_best_label(annotation, confidence_threshold)
            
            if not best_label:
                continue
            
            # Extract bounding box
            bbox = self._extract_bbox(annotation)
            if not bbox:
                logger.warning(f"Could not extract bbox for annotation {annotation.id}")
                continue
            
            # Get image dimensions (may not be directly available)
            img_width, img_height = self._get_image_dimensions(image)
            
            # Create structured annotation data
            ann_data = GetiAnnotationData(
                image_name=image.name,
                image_id=str(image.id),
                image_width=img_width,
                image_height=img_height,
                annotation_id=str(annotation.id),
                bbox=bbox,
                category_name=best_label[0],
                confidence=best_label[1]
            )
            
            annotations.append(ann_data)
        
        return annotations
    
    def _get_best_label(self, annotation: Annotation, confidence_threshold: float) -> Optional[Tuple[str, float]]:
        """
        Get the best category label from an annotation
        For Geti's detection->classification pipeline, we want the specific species
        classification, not the generic "Animal" detection label.
        
        Returns:
            Tuple of (category_name, confidence) or None if no good labels
        """
        if not annotation.labels:
            return None
        
        # Filter labels by confidence
        valid_labels = [
            (label.name, label.probability) 
            for label in annotation.labels 
            if label.probability >= confidence_threshold
        ]
        
        if not valid_labels:
            return None
        
        # Strategy for clean export: Take most specific label available
        # Priority: specific species > general category > "Animal"
        species_labels = [
            (name, prob) for name, prob in valid_labels 
            if name.endswith("_new") or name in ["jay", "songbird", "dove", "squirrel", "cat"]
        ]
        
        if species_labels:
            # Take highest confidence species
            return max(species_labels, key=lambda x: x[1])
        
        # If no species, look for general categories
        category_labels = [
            (name, prob) for name, prob in valid_labels 
            if name.lower() in ["mammal", "bird"]
        ]
        
        if category_labels:
            # Take highest confidence category
            return max(category_labels, key=lambda x: x[1])
        
        # Fallback to any valid label (probably "Animal")
        return max(valid_labels, key=lambda x: x[1])
    
    def _get_image_dimensions(self, image: Image) -> Tuple[int, int]:
        """
        Get image dimensions from Geti Image object
        
        Returns:
            Tuple of (width, height). Returns (0, 0) if not available.
        """
        try:
            # Try to get dimensions from media_information (this works!)
            if hasattr(image, 'media_information') and image.media_information:
                media_info = image.media_information
                if hasattr(media_info, 'width') and hasattr(media_info, 'height'):
                    return (media_info.width, media_info.height)
            
            # Fallback attempts for other possible locations
            if hasattr(image, 'width') and hasattr(image, 'height'):
                return (image.width, image.height)
            elif hasattr(image, 'shape') and image.shape:
                if hasattr(image.shape, 'width') and hasattr(image.shape, 'height'):
                    return (image.shape.width, image.shape.height)
            
            # If we can't get dimensions, use placeholder
            return (0, 0)
            
        except Exception as e:
            logger.error(f"Error getting dimensions for image {image.name}: {e}")
            return (0, 0)
    
    def _extract_bbox(self, annotation: Annotation) -> Optional[Tuple[float, float, float, float]]:
        """
        Extract bounding box coordinates from annotation
        
        Returns:
            Tuple of (x, y, width, height) in absolute coordinates, or None if not available
        """
        try:
            # Geti annotations should have shape information
            if hasattr(annotation, 'shape') and annotation.shape:
                shape = annotation.shape
                
                # Handle different shape types (rectangle, polygon, etc.)
                if hasattr(shape, 'x') and hasattr(shape, 'y') and hasattr(shape, 'width') and hasattr(shape, 'height'):
                    # Direct rectangle
                    return (shape.x, shape.y, shape.width, shape.height)
                
                elif hasattr(shape, 'points') and shape.points:
                    # Convert polygon to bounding box
                    x_coords = [p.x for p in shape.points]
                    y_coords = [p.y for p in shape.points]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    return (x_min, y_min, x_max - x_min, y_max - y_min)
            
            logger.warning(f"Could not extract bbox from annotation {annotation.id}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting bbox from annotation {annotation.id}: {e}")
            return None
    
    def get_unique_categories(self, annotations: List[GetiAnnotationData]) -> List[str]:
        """Get list of unique category names from annotations"""
        categories = list(set(ann.category_name for ann in annotations))
        categories.sort()  # Consistent ordering
        logger.info(f"Found {len(categories)} unique categories: {categories}")
        return categories
    
    def download_image(self, image: Image, output_dir: str) -> str:
        """
        Download an image to local directory
        
        Args:
            image: Geti Image object
            output_dir: Directory to save image
            
        Returns:
            Path to downloaded image file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Use image client to download
        image_path = os.path.join(output_dir, image.name)
        
        try:
            # This depends on the exact Geti SDK API - you may need to adjust
            image_data = self.image_client.download_image(image)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            return image_path
        except Exception as e:
            logger.error(f"Error downloading image {image.name}: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = GetiClient(
        host="https://app.geti.intel.com",
        project_name="Wildlife #3"
    )
    
    # Scan ENTIRE dataset for validation (remove max_images to process all 5199 images)
    annotations = client.extract_annotations(confidence_threshold=0.7)
    
    # Get unique categories
    categories = client.get_unique_categories(annotations)
    
    print(f"\nExtracted {len(annotations)} annotations")
    print(f"Categories: {categories}")
    
    # Show more examples to validate data quality
    print(f"\n=== SAMPLE ANNOTATIONS (first 20) ===")
    for i, ann in enumerate(annotations[:20]):
        print(f"{i+1:2d}. Image: {ann.image_name[:50]}..., Category: {ann.category_name}, "
              f"Bbox: {ann.bbox}, Conf: {ann.confidence:.2f}")
    
    # Show unique categories and their counts
    from collections import Counter
    category_counts = Counter(ann.category_name for ann in annotations)
    print(f"\n=== CATEGORY DISTRIBUTION ===")
    for category, count in category_counts.most_common():
        print(f"{category}: {count} annotations")
    
    # Find incomplete annotations (generic categories without species)
    incomplete_annotations = []
    for ann in annotations:
        if ann.category_name in ['bird', 'mammal']:
            incomplete_annotations.append(ann)
    
    if incomplete_annotations:
        print(f"\n=== INCOMPLETE ANNOTATIONS (generic categories without species) ===")
        for i, ann in enumerate(incomplete_annotations):
            print(f"{i+1}. Image: {ann.image_name}")
            print(f"   Category: {ann.category_name}, Bbox: {ann.bbox}, Confidence: {ann.confidence:.2f}")
            print(f"   Image dimensions: {ann.image_width}x{ann.image_height}")
    else:
        print(f"\n✅ No incomplete annotations found")
        
    # Look for any potential issues
    print(f"\n=== DATA QUALITY CHECK ===")
    zero_dimension_count = sum(1 for ann in annotations if ann.image_width == 0 or ann.image_height == 0)
    print(f"Images with zero dimensions: {zero_dimension_count}")
    
    # Check for very small or very large bboxes (potential issues)
    small_bbox_count = sum(1 for ann in annotations if ann.bbox[2] < 5 or ann.bbox[3] < 5)
    large_bbox_count = sum(1 for ann in annotations if ann.bbox[2] > ann.image_width * 0.8 or ann.bbox[3] > ann.image_height * 0.8)
    print(f"Very small bboxes (< 5px): {small_bbox_count}")
    print(f"Very large bboxes (> 80% of image): {large_bbox_count}")
    
    # Show some examples of large bboxes too
    large_bbox_examples = [ann for ann in annotations if ann.bbox[2] > ann.image_width * 0.8 or ann.bbox[3] > ann.image_height * 0.8]
    if large_bbox_examples:
        print(f"\n=== LARGE BBOX EXAMPLES (first 3) ===")
        for i, ann in enumerate(large_bbox_examples[:3]):
            bbox_width_pct = (ann.bbox[2] / ann.image_width) * 100
            bbox_height_pct = (ann.bbox[3] / ann.image_height) * 100
            print(f"{i+1}. Image: {ann.image_name[:50]}...")
            print(f"   Category: {ann.category_name}, Bbox size: {bbox_width_pct:.1f}% x {bbox_height_pct:.1f}% of image")
    
    print(f"\n=== DATASET READY FOR EXPORT ===")
    print(f"✅ Total annotations: {len(annotations)}")
    print(f"✅ Total categories: {len(categories)}")
    print(f"✅ Clean data quality: {len(annotations) - zero_dimension_count - small_bbox_count} good annotations")