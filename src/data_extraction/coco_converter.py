import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Set
from collections import Counter

from geti_client import GetiClient, GetiAnnotationData

logger = logging.getLogger(__name__)


class COCOConverter:
    """Convert Geti annotations to COCO format"""
    
    def __init__(self, output_dir: str = "exported_dataset"):
        """
        Initialize COCO converter
        
        Args:
            output_dir: Directory to save the COCO dataset
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        
        # Category mapping: remove "_new" suffix
        self.category_mapping = {
            "songbird_new": "songbird",
            "squirrel_new": "squirrel", 
            "cat_new": "cat",
            "dove_new": "dove",
            "jay_new": "jay",
            "Animal": "Animal"  # Keep as-is for detection training
        }
        
        # Categories to exclude from export
        self.excluded_categories = {"No object"}
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        logger.info(f"COCO converter initialized. Output: {self.output_dir}")
    
    def convert_to_coco(self, annotations: List[GetiAnnotationData], 
                       download_images: bool = True) -> Dict:
        """
        Convert Geti annotations to COCO format
        
        Args:
            annotations: List of Geti annotation data
            download_images: Whether to download image files
            
        Returns:
            COCO format dictionary
        """
        logger.info("Converting annotations to COCO format...")
        
        # Filter and clean annotations
        clean_annotations = self._filter_annotations(annotations)
        logger.info(f"Filtered {len(annotations)} → {len(clean_annotations)} annotations")
        
        # Build COCO structure
        coco_data = {
            "info": self._create_info(),
            "licenses": self._create_licenses(),
            "categories": self._create_categories(clean_annotations),
            "images": self._create_images(clean_annotations),
            "annotations": self._create_annotations(clean_annotations)
        }
        
        # Download images if requested
        if download_images:
            self._download_images(coco_data["images"])
        
        # Save COCO JSON
        coco_path = os.path.join(self.output_dir, "annotations.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"COCO dataset saved to {coco_path}")
        self._print_summary(coco_data)
        
        return coco_data
    
    def _filter_annotations(self, annotations: List[GetiAnnotationData]) -> List[GetiAnnotationData]:
        """Filter out unwanted annotations and apply category mapping"""
        filtered = []
        
        for ann in annotations:
            # Skip excluded categories
            if ann.category_name in self.excluded_categories:
                continue
            
            # Skip annotations with missing dimensions
            if ann.image_width == 0 or ann.image_height == 0:
                logger.warning(f"Skipping annotation with zero dimensions: {ann.image_name}")
                continue
            
            # Apply category mapping
            mapped_category = self.category_mapping.get(ann.category_name, ann.category_name)
            
            # Create new annotation with mapped category
            filtered_ann = GetiAnnotationData(
                image_name=ann.image_name,
                image_id=ann.image_id,
                image_width=ann.image_width,
                image_height=ann.image_height,
                annotation_id=ann.annotation_id,
                bbox=ann.bbox,
                category_name=mapped_category,
                confidence=ann.confidence
            )
            
            filtered.append(filtered_ann)
        
        return filtered
    
    def _create_info(self) -> Dict:
        """Create COCO info section"""
        return {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "Wildlife dataset exported from Intel Geti platform",
            "contributor": "Geti to COCO Exporter",
            "url": "",
            "date_created": datetime.now().isoformat()
        }
    
    def _create_licenses(self) -> List[Dict]:
        """Create COCO licenses section"""
        return [{
            "id": 1,
            "name": "Custom License",
            "url": ""
        }]
    
    def _create_categories(self, annotations: List[GetiAnnotationData]) -> List[Dict]:
        """Create COCO categories from annotations"""
        # Get unique categories
        category_names = list(set(ann.category_name for ann in annotations))
        category_names.sort()  # Consistent ordering
        
        categories = []
        for i, name in enumerate(category_names, 1):
            # Determine supercategory
            if name in ["songbird", "dove", "jay"]:
                supercategory = "bird"
            elif name in ["squirrel", "cat"]:
                supercategory = "mammal"
            else:
                supercategory = "animal"
            
            categories.append({
                "id": i,
                "name": name,
                "supercategory": supercategory
            })
        
        return categories
    
    def _create_images(self, annotations: List[GetiAnnotationData]) -> List[Dict]:
        """Create COCO images section"""
        # Get unique images
        image_info = {}
        for ann in annotations:
            if ann.image_id not in image_info:
                # Use semantic filename preserving camera/date/frame info
                safe_filename = self._sanitize_filename(ann.image_name) + ".jpg"
                
                # Use consistent image ID
                numeric_id = int(ann.image_id) if ann.image_id.isdigit() else hash(ann.image_id) % (10**9)
                
                image_info[ann.image_id] = {
                    "id": numeric_id,
                    "width": ann.image_width,
                    "height": ann.image_height,
                    "file_name": safe_filename,
                    "license": 1,
                    "date_captured": ""
                }
        
        return list(image_info.values())
    
    def _sanitize_filename(self, filename: str) -> str:
        """Convert Geti image name to filesystem-safe filename while preserving semantic info"""
        # Extract just the meaningful part (remove path prefixes)
        if '/' in filename:
            filename = filename.split('/')[-1]  # Get last part after final /
        
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
    
    def _create_annotations(self, annotations: List[GetiAnnotationData]) -> List[Dict]:
        """Create COCO annotations section"""
        # Create category name to ID mapping
        category_names = list(set(ann.category_name for ann in annotations))
        category_names.sort()
        category_to_id = {name: i+1 for i, name in enumerate(category_names)}
        
        coco_annotations = []
        for i, ann in enumerate(annotations, 1):
            # Convert bbox format: Geti uses (x, y, width, height), COCO uses same
            bbox = list(ann.bbox)
            
            # Calculate area
            area = bbox[2] * bbox[3]
            
            # Use consistent image ID mapping
            image_id = int(ann.image_id) if ann.image_id.isdigit() else hash(ann.image_id) % (10**9)
            
            coco_ann = {
                "id": i,
                "image_id": image_id,
                "category_id": category_to_id[ann.category_name],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": []  # Empty for bounding box annotations
            }
            
            coco_annotations.append(coco_ann)
        
        return coco_annotations
    
    def _download_images(self, coco_images: List[Dict]):
        """Download images to the output directory"""
        logger.info(f"Downloading {len(coco_images)} images...")
        
        # Note: This is a placeholder implementation
        # The actual image download would depend on the Geti SDK's image download capabilities
        # For now, we'll just log what would be downloaded
        
        for i, img_info in enumerate(coco_images):
            if i % 100 == 0:
                logger.info(f"Downloaded {i}/{len(coco_images)} images...")
            
            # TODO: Implement actual image download using GetiClient
            # For now, just create placeholder
            image_path = os.path.join(self.images_dir, img_info["file_name"])
            # client.download_image(image_id, image_path)
        
        logger.warning("Image download not implemented yet - only COCO JSON created")
    
    def _print_summary(self, coco_data: Dict):
        """Print conversion summary"""
        logger.info("\n" + "="*50)
        logger.info("COCO CONVERSION SUMMARY")
        logger.info("="*50)
        
        # Count annotations per category
        category_counts = Counter()
        category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        
        for ann in coco_data["annotations"]:
            category_name = category_id_to_name[ann["category_id"]]
            category_counts[category_name] += 1
        
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"   • Total images: {len(coco_data['images'])}")
        logger.info(f"   • Total annotations: {len(coco_data['annotations'])}")
        logger.info(f"   • Total categories: {len(coco_data['categories'])}")
        
        logger.info(f"\n📈 Category Distribution:")
        for category, count in category_counts.most_common():
            logger.info(f"   • {category}: {count} annotations")
        
        logger.info(f"\n📁 Output Files:")
        logger.info(f"   • COCO JSON: {os.path.join(self.output_dir, 'annotations.json')}")
        logger.info(f"   • Images: {self.images_dir}/")
        
        logger.info(f"\n🎯 Training Usage:")
        logger.info(f"   • Detection: Use all categories → 'Animal' mapping")
        logger.info(f"   • Classification: Use species categories (exclude 'Animal')")


def export_dataset(host: str, project_name: str, output_dir: str = "exported_dataset"):
    """
    Complete pipeline to export Geti dataset to COCO format
    
    Args:
        host: Geti server URL
        project_name: Name of the Geti project
        output_dir: Output directory for the dataset
    """
    logger.info("Starting Geti to COCO export pipeline...")
    
    # Initialize clients
    geti_client = GetiClient(host=host, project_name=project_name)
    coco_converter = COCOConverter(output_dir=output_dir)
    
    # Extract annotations from Geti
    logger.info("Extracting annotations from Geti...")
    annotations = geti_client.extract_annotations(confidence_threshold=0.7)
    
    # Convert to COCO format
    logger.info("Converting to COCO format...")
    coco_data = coco_converter.convert_to_coco(annotations, download_images=False)
    
    logger.info("🎉 Export complete!")
    return coco_data


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Export the dataset
    coco_data = export_dataset(
        host="https://app.geti.intel.com",
        project_name="Wildlife #3",
        output_dir="wildlife_coco_dataset"
    )
    
    print(f"\n✅ Dataset exported successfully!")
    print(f"📁 Output: wildlife_coco_dataset/")
    print(f"📄 COCO JSON: wildlife_coco_dataset/annotations.json")