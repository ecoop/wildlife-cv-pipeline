# Geti to COCO Dataset Exporter

Export annotated datasets from Intel's Geti platform to COCO format for use in computer vision training pipelines.

## Applications

### Core Components

1. **`geti_client.py`** - Core client library for connecting to and interacting with Intel Geti platform APIs.

### Primary Data Processing

2. **`coco_converter.py`** - Converts Geti annotations and hierarchical class structures to standard COCO JSON format.

3. **`image_downloader.py`** - Downloads images and associated metadata from Geti projects to local storage.

4. **`bbox_converter.py`** - Transforms bounding box annotations between Geti and COCO coordinate systems and formats.

### Validation & Visualization

5. **`coco_validation.py`** - Validates exported COCO JSON files for format compliance and data integrity.

6. **`coco_visualization.py`** - Visualizes COCO annotations on images to verify conversion accuracy and data quality.

### Testing

7. **`test_converter.py`** - Unit tests for annotation conversion functionality and edge cases.

8. **`test_download.py`** - Unit tests for image download and metadata extraction processes.

## Current Status

- Basic Geti API integration (in progress)
- Converting hierarchical annotations (Animal → Bird/Mammal → Species)

## Usage

TODO: Add usage instructions

## Requirements

- Python 3.x
- Intel Geti platform access
- Required dependencies (see requirements.txt)