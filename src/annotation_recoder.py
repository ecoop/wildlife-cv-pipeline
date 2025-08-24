import json
import os
import shutil
from PIL import Image
from collections import defaultdict


def setup_recoding_workspace(coco_file_path, images_dir, workspace_dir="annotation_recoding"):
    """
    Create workspace for manual annotation recoding.
    Exports bbox crops with clear naming for easy sorting.
    """
    # Load COCO data
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create workspace structure
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Create category directories
    categories = ['cat', 'dove', 'jay', 'songbird', 'squirrel']
    for category in categories:
        os.makedirs(os.path.join(workspace_dir, category), exist_ok=True)
    
    # Create "unsure" directory for difficult cases
    os.makedirs(os.path.join(workspace_dir, 'unsure'), exist_ok=True)
    
    # Find "Animal" annotations
    categories_dict = {cat['id']: cat for cat in coco_data['categories']}
    animal_cat_id = None
    for cat_id, cat_info in categories_dict.items():
        if cat_info['name'].lower() == 'animal':
            animal_cat_id = cat_id
            break
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Create image ID to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Find and export "Animal" annotations
    animal_annotations = []
    for image_id, anns in annotations_by_image.items():
        category_ids = set(ann['category_id'] for ann in anns)
        if category_ids == {animal_cat_id}:  # Only "Animal" category
            for ann in anns:
                if ann['category_id'] == animal_cat_id:
                    animal_annotations.append({
                        'annotation': ann,
                        'image_id': image_id,
                        'image_info': image_info[image_id]
                    })
    
    print(f"Exporting {len(animal_annotations)} 'Animal' annotations for recoding...")
    
    # Export crops with clear naming
    exported_count = 0
    annotation_mapping = {}  # Maps filename to annotation_id for later lookup
    
    for item in animal_annotations:
        try:
            ann = item['annotation']
            image_id = item['image_id']
            img_info = item['image_info']
            
            # Load original image
            image_path = os.path.join(images_dir, img_info['file_name'])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert('RGB')
            
            # Extract bbox crop
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            crop = image.crop((x, y, x + w, y + h))
            
            # Create descriptive filename
            crop_filename = f"ann_{ann['id']}_img_{image_id}_{img_info['file_name'].split('.')[0]}.png"
            crop_path = os.path.join(workspace_dir, crop_filename)
            
            # Save crop
            crop.save(crop_path)
            annotation_mapping[crop_filename] = ann['id']
            exported_count += 1
            
        except Exception as e:
            print(f"Error processing annotation {ann['id']}: {e}")
            continue
    
    # Save mapping for later use
    mapping_file = os.path.join(workspace_dir, 'annotation_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(annotation_mapping, f, indent=2)
    
    # Create instructions file
    instructions = f"""
ANNOTATION RECODING WORKSPACE
============================

You have {exported_count} animal bbox crops to reclassify.

INSTRUCTIONS:
1. Review each image crop in this directory
2. Drag/move each crop into the correct subdirectory:
   - cat/        (house cats, feral cats)
   - dove/       (mourning doves, other doves) 
   - jay/        (blue jays, other jays)
   - songbird/   (small birds, finches, sparrows, etc.)
   - squirrel/   (gray squirrels, other squirrels)
   - unsure/     (can't determine or doesn't fit categories)

3. Once you've sorted all images, run:
   python src/update_annotations_from_directories.py

TIPS:
- Start with obvious cases (clear cats, obvious squirrels)
- Use 'unsure/' for anything ambiguous
- You can pause and resume - script tracks progress
- Crops are named: ann_[ID]_img_[IMAGE_ID]_[FILENAME].png

CATEGORIES TO CHOOSE FROM:
{categories}
"""
    
    with open(os.path.join(workspace_dir, 'INSTRUCTIONS.txt'), 'w') as f:
        f.write(instructions)
    
    print(f"\n✅ Workspace created: {workspace_dir}/")
    print(f"📁 Exported {exported_count} bbox crops")
    print(f"📋 See {workspace_dir}/INSTRUCTIONS.txt for next steps")
    
    return workspace_dir, exported_count


def update_annotations_from_directories(workspace_dir, original_coco_file, output_coco_file):
    """
    Update COCO annotations based on how user sorted the crops into directories.
    """
    # Load original COCO data
    with open(original_coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Load annotation mapping
    mapping_file = os.path.join(workspace_dir, 'annotation_mapping.json')
    with open(mapping_file, 'r') as f:
        annotation_mapping = json.load(f)
    
    # Create category name to ID mapping
    category_name_to_id = {}
    for cat in coco_data['categories']:
        category_name_to_id[cat['name'].lower()] = cat['id']
    
    print(f"Available categories: {list(category_name_to_id.keys())}")
    
    # Scan directories for sorted crops
    updates = {}  # annotation_id -> new_category_id
    unsure_annotations = []
    
    for category_dir in ['cat', 'dove', 'jay', 'songbird', 'squirrel', 'unsure']:
        dir_path = os.path.join(workspace_dir, category_dir)
        if not os.path.exists(dir_path):
            continue
            
        files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
        print(f"Found {len(files)} crops in {category_dir}/ directory")
        
        for filename in files:
            if filename in annotation_mapping:
                annotation_id = annotation_mapping[filename]
                
                if category_dir == 'unsure':
                    unsure_annotations.append(annotation_id)
                else:
                    if category_dir in category_name_to_id:
                        updates[annotation_id] = category_name_to_id[category_dir]
                    else:
                        print(f"Warning: Unknown category '{category_dir}'")
    
    # Check for unsorted crops (still in root directory)
    unsorted_files = [f for f in os.listdir(workspace_dir) 
                     if f.endswith('.png') and f in annotation_mapping]
    
    print(f"\nRecoding Summary:")
    print(f"  Updates to apply: {len(updates)}")
    print(f"  Marked as unsure: {len(unsure_annotations)}")
    print(f"  Still unsorted: {len(unsorted_files)}")
    
    if unsorted_files:
        print(f"  ⚠️  You have {len(unsorted_files)} unsorted crops remaining!")
        print(f"     Sort these into category directories before proceeding.")
        return False
    
    # Apply updates to COCO data
    updated_count = 0
    removed_count = 0
    
    # Create new annotations list
    new_annotations = []
    for ann in coco_data['annotations']:
        ann_id = ann['id']
        
        if ann_id in updates:
            # Update category
            ann['category_id'] = updates[ann_id]
            new_annotations.append(ann)
            updated_count += 1
        elif ann_id in unsure_annotations:
            # Remove uncertain annotations
            removed_count += 1
            # Don't add to new_annotations
        else:
            # Keep unchanged
            new_annotations.append(ann)
    
    # Update COCO data
    coco_data['annotations'] = new_annotations
    
    # Save updated COCO file
    with open(output_coco_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✅ Updates applied!")
    print(f"📝 Updated {updated_count} annotations")
    print(f"🗑️  Removed {removed_count} uncertain annotations")
    print(f"💾 Saved to: {output_coco_file}")
    
    # Create update summary
    summary = {
        'updated_annotations': updated_count,
        'removed_annotations': removed_count,
        'category_updates': {}
    }
    
    # Count updates per category
    for ann_id, new_cat_id in updates.items():
        cat_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == new_cat_id)
        summary['category_updates'][cat_name] = summary['category_updates'].get(cat_name, 0) + 1
    
    print(f"\nUpdates by category:")
    for cat_name, count in summary['category_updates'].items():
        print(f"  {cat_name}: {count}")
    
    return True


def main():
    """
    Main recoding workflow
    """
    print("=== ANNOTATION RECODING SYSTEM ===\n")
    
    # Configuration
    coco_file = 'wildlife_coco_dataset/annotations.json'
    images_dir = 'wildlife_coco_dataset/images'  # Adjust as needed
    workspace_dir = 'annotation_recoding'
    
    # Check if workspace already exists
    if os.path.exists(workspace_dir):
        response = input(f"Workspace '{workspace_dir}' exists. Continue with recoding? (y/n): ")
        if response.lower() == 'y':
            # Apply updates
            output_file = 'wildlife_coco_dataset/annotations_updated.json'
            success = update_annotations_from_directories(workspace_dir, coco_file, output_file)
            if success:
                print(f"\n🎉 Recoding complete! Use {output_file} for training.")
        else:
            print("Setup cancelled.")
    else:
        # Setup workspace
        setup_recoding_workspace(coco_file, images_dir, workspace_dir)
        print(f"\n📋 Next steps:")
        print(f"1. Open the '{workspace_dir}' directory")
        print(f"2. Sort the image crops into category subdirectories")
        print(f"3. Run this script again to apply updates")


if __name__ == "__main__":
    main()