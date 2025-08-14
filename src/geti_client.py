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
        logger.info("Fetching all images...")
        images = self.image_client.get_all_images()
        logger.info(f"Found {len(images)} images")
        return images
    
    def extract_annotations(self, confidence_threshold: float = 0.5) -> List[GetiAnnotationData]:
        """
        Extract all annotations from the project
        
        Args:
            confidence_threshold: Minimum confidence for including annotations
            
        Returns:
            List of structured annotation data
        """
        images = self.get_all_images()
        all_annotations = []
        
        logger.info("Extracting annotations...")
        
        for i, image in enumerate(images):
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
                
            except Exception as e:
                logger.error(f"Error processing image {image.name}: {e}")
                continue
        
        logger.info(f"Extracted {len(all_annotations)} annotations total")
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
            
            # Create structured annotation data
            ann_data = GetiAnnotationData(
                image_name=image.name,
                image_id=str(image.id),
                image_width=image.width,
                image_height=image.height,
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
        
        Returns:
            Tuple of (category_name, confidence) or None if no good labels
        """
        if not annotation.labels:
            return None
        
        # Filter labels by confidence and find the best one
        valid_labels = [
            (label.name, label.probability) 
            for label in annotation.labels 
            if label.probability >= confidence_threshold
        ]
        
        if not valid_labels:
            return None
        
        # Return label with highest confidence
        return max(valid_labels, key=lambda x: x[1])
    
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
    
    # Extract all annotations
    annotations = client.extract_annotations(confidence_threshold=0.7)
    
    # Get unique categories
    categories = client.get_unique_categories(annotations)
    
    print(f"\nExtracted {len(annotations)} annotations")
    print(f"Categories: {categories}")
    
    # Show some sample annotations
    for ann in annotations[:5]:
        print(f"Image: {ann.image_name}, Category: {ann.category_name}, "
              f"Bbox: {ann.bbox}, Confidence: {ann.confidence:.2f}")
