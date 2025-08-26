import json
import os
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict


def export_animal_review_images(coco_file_path, images_dir, output_dir="animal_review"):
    """
    Export images that only have 'Animal' annotations for manual review.
    
    Args:
        coco_file_path: Path to COCO annotations.json
        images_dir: Directory containing the original images
        output_dir: Directory to save review images
    """
    # Load COCO data
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    bbox_crops_dir = os.path.join(output_dir, "bbox_crops")
    os.makedirs(bbox_crops_dir, exist_ok=True)
    
    # Get category mapping
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # Find "Animal" category ID
    animal_cat_id = None
    for cat_id, cat_info in categories.items():
        if cat_info['name'].lower() == 'animal':
            animal_cat_id = cat_id
            break
    
    if animal_cat_id is None:
        print("No 'Animal' category found!")
        return
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Create image ID to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Find images with ONLY "Animal" annotations
    animal_only_images = []
    for image_id, anns in annotations_by_image.items():
        category_ids = set(ann['category_id'] for ann in anns)
        if category_ids == {animal_cat_id}:  # Only "Animal" category
            animal_only_images.append(image_id)
    
    print(f"Found {len(animal_only_images)} images with only 'Animal' annotations")
    print(f"Exporting to: {output_dir}/")
    
    # Process each animal-only image
    bbox_count = 0
    processed_count = 0
    
    for i, image_id in enumerate(animal_only_images):
        try:
            # Get image info and annotations
            img_info = image_info[image_id]
            image_path = os.path.join(images_dir, img_info['file_name'])
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # Get annotations for this image
            image_annotations = annotations_by_image[image_id]
            animal_annotations = [ann for ann in image_annotations if ann['category_id'] == animal_cat_id]
            
            # Draw bounding boxes
            for j, ann in enumerate(animal_annotations):
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # Draw rectangle
                draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
                
                # Add annotation ID label
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                draw.text((x, y - 20), f"Animal_{ann['id']}", fill='red', font=font)
                
                # Also save individual bbox crops
                crop = image.crop((x, y, x + w, y + h))
                crop_filename = f"animal_{image_id}_{ann['id']}.png"
                crop.save(os.path.join(bbox_crops_dir, crop_filename))
                bbox_count += 1
            
            # Save full image with annotations
            output_filename = f"review_{image_id}_{img_info['file_name']}"
            output_path = os.path.join(output_dir, output_filename)
            image.save(output_path)
            
            processed_count += 1
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(animal_only_images)} images...")
                
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue
    
    print(f"\n✅ Export complete!")
    print(f"Processed: {processed_count} images")
    print(f"Exported: {bbox_count} individual bbox crops")
    print(f"Full images with bboxes: {output_dir}/")
    print(f"Individual bbox crops: {bbox_crops_dir}/")
    
    # Create summary file
    summary_path = os.path.join(output_dir, "review_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Animal Review Summary\n")
        f.write(f"===================\n")
        f.write(f"Total images with only 'Animal' annotations: {len(animal_only_images)}\n")
        f.write(f"Successfully processed: {processed_count}\n")
        f.write(f"Total 'Animal' bboxes: {bbox_count}\n")
        f.write(f"\nImage IDs processed:\n")
        for img_id in animal_only_images[:20]:  # First 20
            f.write(f"  {img_id}\n")
        if len(animal_only_images) > 20:
            f.write(f"  ... and {len(animal_only_images) - 20} more\n")
    
    return {
        'processed_images': processed_count,
        'total_bboxes': bbox_count,
        'output_dir': output_dir
    }


def main():
    """
    Main function to export animal review images.
    """
    coco_file = 'wildlife_coco_dataset/annotations.json'
    images_directory = 'wildlife_coco_dataset/images'  # Adjust this path as needed
    
    print("Starting animal review export...")
    results = export_animal_review_images(coco_file, images_directory)
    
    print(f"\nNext steps:")
    print(f"1. Review images in: {results['output_dir']}/")
    print(f"2. Check individual crops in: {results['output_dir']}/bbox_crops/")
    print(f"3. Identify what animals these are (cats, birds, squirrels, etc.)")
    print(f"4. Go back to Geti to correct the annotations")
    print(f"5. Re-export the dataset with corrected annotations")


if __name__ == "__main__":
    main()