import os
from typing import Generic, List, Optional, Sequence, Union

from geti_sdk import Geti
from geti_sdk.rest_clients import ProjectClient, ImageClient, AnnotationClient
from geti_sdk.data_models import AnnotationScene, Image


# === CONFIG ===
HOST = "https://app.geti.intel.com"
PROJECT_NAME = "Wildlife #3"
ATTRIBUTE_MAPPING = {
    "scrub jay": "jay_new",
    "songbird": "songbird_new",
    "dove": "dove_new",
    "other bird": "other_bird_new",
    "cat": "cat_new",
    "squirrel": "squirrel_new",
    "mammal": "other_mammal_new",
    "other mammal": "other_mammal_new"
}

# === AUTH ===
API_TOKEN = os.getenv("GETI_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("GETI_API_TOKEN not set in environment")

print(f"\n API_TOKEN: '{API_TOKEN}'")

# === CONNECT TO GETI ===
geti = Geti(host=HOST, token=API_TOKEN)

project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)

projects = project_client.list_projects()

project = geti.get_project(PROJECT_NAME)
if not project:
    raise ValueError(f"Project '{PROJECT_NAME}' not found")


violations = []
updates = []

# ALL GOOD TO HERE ... BUT WHAT NEXT? =ec 25.07.08

image_client = ImageClient(
    session=geti.session, workspace_id=geti.workspace_id, project=project
)

images = image_client.get_all_images()
print(f"Project '{project.name}' contains {len(images)} images")

annotation_client = AnnotationClient(
    session=geti.session, workspace_id=geti.workspace_id, project=project
)

# === DRY-RUN MAIN LOGIC ===
debug_ndx = -1
for image in images:
    annotationScene = annotation_client.get_annotation(media_item=image)
    image_modified = False
    modified_annotations = []

    debug_ndx += 1
    if not isinstance(annotationScene, AnnotationScene):
        continue

    for index, ann in enumerate(annotationScene.annotations):

        debug_ndx += 1

        # Get current label names with high confidence
        true_keys = [label.name for label in ann.labels if label.probability == 1.0]

        # Flag violations where there are more than 3 true keys
        if len(true_keys) > 3: 
            violations.append({
                "image": image.name,
                "annotation_id": ann.id,
                "true_keys": true_keys
            })

        # Check for rename-needed labels
        # renamed_labels = []
        # for label in ann.labels:
        #     new_name = ATTRIBUTE_MAPPING.get(label.name)
        #     if new_name and new_name != label.name:
        #         renamed_labels.append((label.name, new_name))

        # if renamed_labels:
        #     image_modified = True
        #     modified_annotations.append({
        #         "annotation_id": ann.id,
        #         "renamed_labels": renamed_labels,
        #         "label_names_before": [label.name for label in ann.labels],
        #         "label_names_after": [
        #             ATTRIBUTE_MAPPING.get(label.name, label.name) for label in ann.labels
        #         ]
        #     })

    if image_modified:
        updates.append({
            "image": image.name,
            "annotations": modified_annotations
        })


    # if debug_ndx >= 1000:
    #     break # Exit loop after 100 images

# === REPORT DRY-RUN RESULTS ===
print("\n====================")
print("DRY-RUN COMPLETE")
print("====================")

# if updates:
#     print(f"\n🔁 {len(updates)} image(s) would be updated:")
#     for u in updates:
#         print(f" - {u['image']}")
#         for ann in u["annotations"]:
#             print(f"    • Annotation ID: {ann['annotation_id']}")
#             for old, new in ann["renamed_labels"]:
#                 print(f"      Rename: {old} → {new}")
#             print(f"      Labels before: {ann['label_names_before']}")
#             print(f"      Labels after:  {ann['label_names_after']}")
# else:
#     print("\n✅ No label renames needed.")


if violations:
    print(f"\n⚠️  {len(violations)} annotation(s) have multiple true attributes:")
    for v in violations:
        print(f" - Image: {v['image']} | Ann ID: {v['annotation_id']} | True keys: {v['true_keys']}")
else:
    print("\n✅ No multiple-true-attribute violations found.")
