import json

# Load and inspect the COCO file
with open('wildlife_coco_dataset/annotations.json', 'r') as f:
    coco = json.load(f)

# Sanity checks
print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
print(f"Categories: {[cat['name'] for cat in coco['categories']]}")

# Verify cleanup worked
categories = [cat['name'] for cat in coco['categories']]
assert "No object" not in categories, "❌ 'No object' not removed"
assert not any("_new" in cat for cat in categories), "❌ '_new' not removed"
print("✅ Category cleanup successful")

# Check bbox sanity (first 100 annotations)
for ann in coco['annotations'][:100]:
    bbox = ann['bbox']
    print(f"Bbox: {bbox} (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})")