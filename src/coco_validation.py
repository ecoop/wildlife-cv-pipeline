import json
from collections import defaultdict


def analyze_hierarchy_issues(coco_data):
    """
    Analyze hierarchy and category issues in COCO dataset.
    
    Args:
        coco_data: Loaded COCO JSON data
        
    Returns:
        Dict with analysis results
    """
    # Get category mapping
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # Find generic "Animal" category ID
    animal_cat_id = None
    for cat_id, cat_info in categories.items():
        if cat_info['name'].lower() == 'animal':
            animal_cat_id = cat_id
            break
    
    print(f"Animal category ID: {animal_cat_id}")
    
    # Analyze annotations by category
    annotations_by_category = defaultdict(list)
    annotations_by_image = defaultdict(list)
    
    for ann in coco_data['annotations']:
        annotations_by_category[ann['category_id']].append(ann)
        annotations_by_image[ann['image_id']].append(ann)
    
    # 1. Find images with only "Animal" annotations
    animal_only_images = []
    mixed_images = []  # Images with both "Animal" and specific categories
    
    for image_id, anns in annotations_by_image.items():
        category_ids = set(ann['category_id'] for ann in anns)
        
        if animal_cat_id in category_ids:
            if len(category_ids) == 1:  # Only "Animal"
                animal_only_images.append(image_id)
            else:  # "Animal" + other categories
                mixed_images.append({
                    'image_id': image_id,
                    'categories': [categories[cid]['name'] for cid in category_ids],
                    'annotations': anns
                })
    
    # 2. Check for overlapping bounding boxes in mixed images
    def bbox_overlap(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        area1, area2 = w1 * h1, w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    overlapping_annotations = []
    for mixed_img in mixed_images:
        anns = mixed_img['annotations']
        for i in range(len(anns)):
            for j in range(i+1, len(anns)):
                overlap = bbox_overlap(anns[i]['bbox'], anns[j]['bbox'])
                if overlap > 0.7:  # High overlap threshold
                    overlapping_annotations.append({
                        'image_id': mixed_img['image_id'],
                        'ann1': anns[i],
                        'ann2': anns[j],
                        'overlap': overlap,
                        'cat1': categories[anns[i]['category_id']]['name'],
                        'cat2': categories[anns[j]['category_id']]['name']
                    })
    
    # 3. Summary report
    results = {
        'total_animal_annotations': len(annotations_by_category[animal_cat_id]) if animal_cat_id else 0,
        'animal_only_images': len(animal_only_images),
        'mixed_images': len(mixed_images),
        'overlapping_annotations': len(overlapping_annotations),
        'category_distribution': {
            categories[cat_id]['name']: len(anns) 
            for cat_id, anns in annotations_by_category.items()
        }
    }
    
    # Print detailed analysis
    print("\n=== HIERARCHY ANALYSIS ===")
    print(f"Total 'Animal' annotations: {results['total_animal_annotations']}")
    print(f"Images with ONLY 'Animal' category: {results['animal_only_images']}")
    print(f"Images with 'Animal' + other categories: {results['mixed_images']}")
    print(f"Potentially duplicate annotations (high bbox overlap): {results['overlapping_annotations']}")
    
    print("\n=== CATEGORY DISTRIBUTION ===")
    for cat_name, count in results['category_distribution'].items():
        print(f"{cat_name}: {count}")
    
    # Sample problematic cases
    if overlapping_annotations:
        print(f"\n=== SAMPLE OVERLAPPING ANNOTATIONS ===")
        for i, overlap_case in enumerate(overlapping_annotations[:5]):
            print(f"Image {overlap_case['image_id']}: {overlap_case['cat1']} vs {overlap_case['cat2']} "
                  f"(overlap: {overlap_case['overlap']:.2f})")
    
    if animal_only_images:
        print(f"\n=== SAMPLE 'ANIMAL-ONLY' IMAGES ===")
        print(f"First 10 image IDs: {animal_only_images[:10]}")
    
    return results


def suggest_cleanup_strategy(analysis_results):
    """
    Suggest cleanup strategies based on analysis results.
    """
    print("\n=== CLEANUP RECOMMENDATIONS ===")
    
    if analysis_results['overlapping_annotations'] > 0:
        print("1. OVERLAPPING ANNOTATIONS:")
        print("   - Likely duplicate annotations at different hierarchy levels")
        print("   - Recommend: Remove generic 'Animal' annotations when specific subcategory exists")
    
    if analysis_results['animal_only_images'] > 0:
        print("2. GENERIC 'ANIMAL' ANNOTATIONS:")
        print("   - These need manual review or re-classification")
        print("   - Options: a) Re-export from Geti with better category mapping")
        print("            b) Manual review and re-labeling")
        print("            c) Exclude from training dataset")
    
    print("3. RECOMMENDED NEXT STEPS:")
    print("   - Visualize sample 'animal-only' images to understand what they contain")
    print("   - Check if Geti export settings can be improved")
    print("   - Consider creating a 'cleanup' function to remove redundant annotations")


def main():
    # Load and inspect the COCO file
    with open('wildlife_coco_dataset/annotations.json', 'r') as f:
        coco = json.load(f)

    # Original sanity checks
    print("=== BASIC VALIDATION ===")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in coco['categories']]}")

    # Verify cleanup worked
    categories = [cat['name'] for cat in coco['categories']]
    assert "No object" not in categories, "❌ 'No object' not removed"
    assert not any("_new" in cat for cat in categories), "❌ '_new' not removed"
    print("✅ Category cleanup successful")

    # Check bbox sanity (first 100 annotations)
    print("\n=== BBOX SAMPLE (first 5) ===")
    for ann in coco['annotations'][:5]:
        bbox = ann['bbox']
        print(f"Bbox: {bbox} (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})")

    # New hierarchy analysis
    results = analyze_hierarchy_issues(coco)
    suggest_cleanup_strategy(results)


if __name__ == "__main__":
    main()