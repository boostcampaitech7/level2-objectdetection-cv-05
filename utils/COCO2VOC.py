import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import argparse

def create_pascal_voc_xml(image_info, annotations, categories, output_dir):
    root = ET.Element("annotation")
    
    ET.SubElement(root, "folder").text = "VOC2012"
    ET.SubElement(root, "filename").text = image_info['file_name']
    
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image_info['width'])
    ET.SubElement(size, "height").text = str(image_info['height'])
    ET.SubElement(size, "depth").text = "3"
    
    ET.SubElement(root, "segmented").text = "0"
    
    for ann in annotations:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = categories[ann['category_id']]['name']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(ann['bbox'][0]))
        ET.SubElement(bbox, "ymin").text = str(int(ann['bbox'][1]))
        ET.SubElement(bbox, "xmax").text = str(int(ann['bbox'][0] + ann['bbox'][2]))
        ET.SubElement(bbox, "ymax").text = str(int(ann['bbox'][1] + ann['bbox'][3]))
    
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    
    with open(os.path.join(output_dir, os.path.splitext(image_info['file_name'])[0] + ".xml"), "w") as f:
        f.write(xml_str)

def convert_coco_to_pascal(coco_json, output_dir):
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_dict = {img['id']: img for img in coco_data['images']}
    
    for img_id, img_info in image_dict.items():
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        create_pascal_voc_xml(img_info, annotations, {cat['id']: cat for cat in coco_data['categories']}, output_dir)
    
    print(f"Conversion complete. Pascal VOC XML files saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert COCO JSON to Pascal VOC XML")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to COCO JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for Pascal VOC XML files")
    
    args = parser.parse_args()
    
    convert_coco_to_pascal(args.coco_json, args.output_dir)

if __name__ == "__main__":
    main()