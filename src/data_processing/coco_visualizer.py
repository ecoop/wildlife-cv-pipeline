import os
import json
import random
import logging
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np

logger = logging.getLogger(__name__)


class COCOVisualizer:
    """Visualize COCO dataset with bounding boxes and labels"""
    
    def __init__(self, coco_json_path: str, images_dir: str):
        """
        Initialize COCO visualizer
        
        Args:
            coco_json_path: Path to COCO annotations.json
            images_dir: Directory containing the images
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        
        # Load COCO data
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create lookup mappings
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)
        
        # Define colors for different categories
        self.category_colors = self._generate_category_colors()
        
        logger.info(f"Loaded COCO dataset:")
        logger.info(f"  Images: {len(self.coco_data['images'])}")
        logger.info(f"  Annotations: {len(self.coco_data['annotations'])}")
        logger.info(f"  Categories: {len(self.coco_data['categories'])}")
    
    def _generate_category_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Generate distinct colors for each category"""
        colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green  
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
            (255, 192, 203), # Pink
            (165, 42, 42),   # Brown
        ]
        
        category_colors = {}
        for i, category in enumerate(self.coco_data['categories']):
            color_idx = i % len(colors)
            category_colors[category['name']] = colors[color_idx]
        
        return category_colors
    
    def visualize_single_image(self, image_id: int, output_path: Optional[str] = None, 
                              show_confidence: bool = True, font_size: int = 12) -> bool:
        """
        Visualize a single image with its annotations
        
        Args:
            image_id: COCO image ID to visualize
            output_path: Path to save annotated image (optional)
            show_confidence: Whether to show confidence scores in labels
            font_size: Font size for labels
            
        Returns:
            True if successful, False otherwise
        """
        if image_id not in self.image_id_to_info:
            logger.error(f"Image ID {image_id} not found in COCO data")
            return False
        
        # Get image info
        image_info = self.image_id_to_info[image_id]
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return False
        
        # Create drawing context
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Draw annotations
        annotations = self.image_annotations.get(image_id, [])
        logger.info(f"Drawing {len(annotations)} annotations on {image_info['file_name']}")
        
        for ann in annotations:
            # Get category info
            category_name = self.category_id_to_name[ann['category_id']]
            color = self.category_colors[category_name]
            
            # Get bbox coordinates (COCO format: x, y, width, height)
            x, y, width, height = ann['bbox']
            
            # Draw bounding box
            draw.rectangle(
                [(x, y), (x + width, y + height)],
                outline=color,
                width=3
            )
            
            # Create label text
            label_text = category_name
            if show_confidence and 'confidence' in ann:
                label_text += f" ({ann['confidence']:.2f})"
            
            # Draw label background
            bbox_label = draw.textbbox((0, 0), label_text, font=font)
            label_width = bbox_label[2] - bbox_label[0]
            label_height = bbox_label[3] - bbox_label[1]
            
            # Position label above bbox, or below if near top
            label_y = y - label_height - 2 if y > label_height + 10 else y + height + 2
            label_x = x
            
            # Draw label background
            draw.rectangle(
                [(label_x, label_y), (label_x + label_width + 4, label_y + label_height + 4)],
                fill=color,
                outline=color
            )
            
            # Draw label text
            draw.text((label_x + 2, label_y + 2), label_text, fill='white', font=font)
        
        # Save or show image
        if output_path:
            image.save(output_path, 'JPEG', quality=95)
            logger.info(f"Saved annotated image to {output_path}")
        else:
            # Display using matplotlib
            plt.figure(figsize=(12, 8))
            plt.imshow(np.array(image))
            plt.axis('off')
            plt.title(f"{image_info['file_name']} ({len(annotations)} annotations)")
            plt.tight_layout()
            plt.show()
        
        return True
    
    def create_sample_grid(self, num_samples: int = 9, output_path: str = "sample_grid.jpg",
                          grid_size: Tuple[int, int] = (3, 3)) -> bool:
        """
        Create a grid of sample images with annotations
        
        Args:
            num_samples: Number of sample images to include
            output_path: Path to save the grid image
            grid_size: Grid dimensions (rows, cols)
            
        Returns:
            True if successful, False otherwise
        """
        # Get random sample of images with annotations
        images_with_annotations = [
            img_id for img_id in self.image_annotations.keys()
            if len(self.image_annotations[img_id]) > 0
        ]
        
        if len(images_with_annotations) < num_samples:
            logger.warning(f"Only {len(images_with_annotations)} images with annotations available")
            num_samples = len(images_with_annotations)
        
        sample_image_ids = random.sample(images_with_annotations, num_samples)
        
        # Create matplotlib grid
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i, image_id in enumerate(sample_image_ids):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get image info and load image
            image_info = self.image_id_to_info[image_id]
            image_path = os.path.join(self.images_dir, image_info['file_name'])
            
            try:
                image = Image.open(image_path).convert('RGB')
                ax.imshow(np.array(image))
                
                # Draw bounding boxes
                annotations = self.image_annotations[image_id]
                for ann in annotations:
                    category_name = self.category_id_to_name[ann['category_id']]
                    color = [c/255.0 for c in self.category_colors[category_name]]  # Normalize to [0,1]
                    
                    x, y, width, height = ann['bbox']
                    rect = Rectangle((x, y), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(x, y-5, category_name, color=color, fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Set title and remove axes
                filename = image_info['file_name'][:30] + "..." if len(image_info['file_name']) > 30 else image_info['file_name']
                ax.set_title(f"{filename}\n({len(annotations)} annotations)", fontsize=10)
                ax.axis('off')
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                ax.text(0.5, 0.5, f"Error loading\n{image_info['file_name']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(sample_image_ids), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Sample grid saved to {output_path}")
        return True
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset"""
        # Count annotations per category
        category_counts = {}
        for ann in self.coco_data['annotations']:
            cat_name = self.category_id_to_name[ann['category_id']]
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        # Count images per category
        images_per_category = {}
        for image_id, annotations in self.image_annotations.items():
            categories_in_image = set()
            for ann in annotations:
                cat_name = self.category_id_to_name[ann['category_id']]
                categories_in_image.add(cat_name)
            
            for cat_name in categories_in_image:
                images_per_category[cat_name] = images_per_category.get(cat_name, 0) + 1
        
        # Calculate bbox size statistics
        bbox_areas = []
        for ann in self.coco_data['annotations']:
            bbox_areas.append(ann['area'])
        
        stats = {
            'total_images': len(self.coco_data['images']),
            'total_annotations': len(self.coco_data['annotations']),
            'total_categories': len(self.coco_data['categories']),
            'annotations_per_category': category_counts,
            'images_per_category': images_per_category,
            'avg_annotations_per_image': len(self.coco_data['annotations']) / len(self.coco_data['images']),
            'bbox_area_stats': {
                'mean': np.mean(bbox_areas),
                'median': np.median(bbox_areas),
                'min': np.min(bbox_areas),
                'max': np.max(bbox_areas)
            }
        }
        
        return stats
    
    def print_dataset_summary(self):
        """Print a summary of the dataset"""
        stats = self.get_dataset_stats()
        
        print("\n" + "="*50)
        print("COCO DATASET SUMMARY")
        print("="*50)
        print(f"📊 Total Images: {stats['total_images']}")
        print(f"📋 Total Annotations: {stats['total_annotations']}")
        print(f"🏷️  Total Categories: {stats['total_categories']}")
        print(f"📈 Avg Annotations/Image: {stats['avg_annotations_per_image']:.1f}")
        
        print(f"\n📋 Annotations per Category:")
        for category, count in sorted(stats['annotations_per_category'].items(), 
                                    key=lambda x: x[1], reverse=True):
            print(f"   • {category}: {count}")
        
        print(f"\n🖼️  Images per Category:")
        for category, count in sorted(stats['images_per_category'].items(), 
                                    key=lambda x: x[1], reverse=True):
            print(f"   • {category}: {count}")
        
        bbox_stats = stats['bbox_area_stats']
        print(f"\n📐 Bounding Box Area Statistics:")
        print(f"   • Mean: {bbox_stats['mean']:.0f} pixels²")
        print(f"   • Median: {bbox_stats['median']:.0f} pixels²")
        print(f"   • Range: {bbox_stats['min']:.0f} - {bbox_stats['max']:.0f} pixels²")


def visualize_coco_dataset(coco_json_path: str, images_dir: str, 
                          create_grid: bool = True, show_samples: int = 3):
    """
    Main function to visualize COCO dataset
    
    Args:
        coco_json_path: Path to COCO annotations.json
        images_dir: Directory containing images
        create_grid: Whether to create sample grid
        show_samples: Number of individual samples to show
    """
    # Initialize visualizer
    visualizer = COCOVisualizer(coco_json_path, images_dir)
    
    # Print dataset summary
    visualizer.print_dataset_summary()
    
    # Create sample grid
    if create_grid:
        print(f"\n🖼️  Creating sample grid...")
        visualizer.create_sample_grid(num_samples=9, output_path="dataset_sample_grid.jpg")
    
    # Show individual samples
    if show_samples > 0:
        print(f"\n🔍 Showing {show_samples} individual samples...")
        
        # Get random images with annotations
        images_with_annotations = [
            img_id for img_id in visualizer.image_annotations.keys()
            if len(visualizer.image_annotations[img_id]) > 0
        ]
        
        sample_ids = random.sample(images_with_annotations, min(show_samples, len(images_with_annotations)))
        
        for i, image_id in enumerate(sample_ids):
            print(f"Visualizing sample {i+1}/{len(sample_ids)}")
            visualizer.visualize_single_image(image_id, output_path=f"sample_{i+1}.jpg")


# Example usage
if __name__ == "__main__":
    # Visualize your COCO dataset
    visualize_coco_dataset(
        coco_json_path="wildlife_coco_dataset/annotations.json",
        images_dir="wildlife_coco_dataset/images",
        create_grid=True,
        show_samples=3
    )
    
    print("\n✅ Visualization complete!")
    print("📁 Check the following files:")
    print("   • dataset_sample_grid.jpg - Overview grid")
    print("   • sample_1.jpg, sample_2.jpg, etc. - Individual samples")