import os
import glob
import json
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List
from collections import defaultdict

def get_annotations(xml_file: str) -> Dict:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin]
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }

def voc_to_coco(VOC_xml: str, output_dir: str):
    xml_files = glob.glob(os.path.join(VOC_xml, '*.xml'))
    
    categories_set = set()
    annotations = []
    images = []
    
    for xml_file in xml_files:
        ann = get_annotations(xml_file)
        
        # 이미지 id와 file_name이 일치하도록 수정
        file_name = os.path.splitext(ann['filename'])[0]
        image_id = int(file_name.split('/')[-1].split('.')[0])  # 파일 이름에서 번호를 추출하여 id로 사용
        images.append({
            'id': image_id,
            'file_name': ann['filename'],
            'width': ann['width'],
            'height': ann['height'],
            'license': 0,
            'flickr_url': None,
            'coco_url': None,
            'date_captured': "2020-12-26 14:44:23"
        })
        
        for obj in ann['objects']:
            if obj['name'] not in categories_set:
                categories_set.add(obj['name'])
            
            category_id = list(categories_set).index(obj['name'])
            
            annotations.append({
                'id': len(annotations),
                'image_id': image_id,
                'category_id': category_id,
                'bbox': obj['bbox'],
                'area': obj['bbox'][2] * obj['bbox'][3],
                'iscrowd': 0
            })
    
    categories = [{'id': i, 'name': cat, 'supercategory': cat} for i, cat in enumerate(categories_set)]
    
    coco_format = {
        'info': {
            'year': 2021,
            'version': '1.0',
            'description': 'Recycle Trash',
            'contributor': 'Upstage',
            'url': None,
            'date_created': '2021-02-02 01:10:00'
        },
        'licenses': [
            {
                'id': 0,
                'name': 'CC BY 4.0',
                'url': 'https://creativecommons.org/licenses/by/4.0/deed.ast'
            }
        ],
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    
    with open(output_dir, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"Conversion completed. COCO format annotations saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert VOC format annotations to COCO format.")
    parser.add_argument('--VOC_xml', required=True, help="Directory containing VOC XML files")
    parser.add_argument('--output_dir', required=True, help="Output JSON file for COCO format")
    args = parser.parse_args()

    voc_to_coco(args.VOC_xml, args.output_dir)
