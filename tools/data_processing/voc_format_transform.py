import os
import glob as glob
import argparse
import yaml
import xml.etree.ElementTree as ET

def findXmlText(tag, txt_string):
    output = []
    for x in tag.findall(txt_string):
        if x.text != None:
            return x.text
    

def vocfile_to_yolov1(voc_path : str, output_path : str, label_list : dict):
    tree = ET.parse(voc_path)
    root = tree.getroot()
        
    yololines = []
    for x in root:
        if x.tag == 'filename':
            image_file_name = x.text

        if x.tag == 'size':
            width = float(findXmlText(x, 'width'))
            height = float(findXmlText(x, 'height'))
        
        if x.tag == 'object':
            classification = findXmlText(x, 'name')
            print(classification)
            for y in x:
                if y.tag == 'bndbox':
                    xmin = float(findXmlText(y, 'xmin'))
                    ymin = float(findXmlText(y, 'ymin'))
                    xmax = float(findXmlText(y, 'xmax'))
                    ymax = float(findXmlText(y, 'ymax'))

            assert classification in label_list, f'{classification} is not in your config, please check `label_mapping_rev` in the config again'
            cls_index = str(label_list[classification])
            center_x = (xmin + (xmax-xmin)/2) / width
            center_y = (ymin + (ymax-ymin)/2) / height
            box_width = (xmax - xmin)/width
            box_height = (ymax - ymin)/height

            yololines.append(' '.join([str(cls_index), str(f'{center_x:.6f}'), str(f'{center_y:.6f}'), str(f'{box_width:.6f}'), str(f'{box_height:.6f}')]))

    yolo_label = '\n'.join(yololines).strip()
    output_file_path = os.path.join(output_path, f"{os.path.basename(voc_path).split('.xml')[0]}.txt")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_file_path, 'w') as f:
        f.write(yolo_label)

def vocdataset_to_yolov1(input_xml : str, output_folder : str, label_list : dict):
    if input_xml.split('.')[-1] == 'xml':
        vocfile_to_yolov1(input_xml, output_folder, label_list)
    else:
        voc_files = glob.glob(f'{input_xml}/*xml')
        for voc_file in voc_files:
            vocfile_to_yolov1(voc_file, output_folder, label_list)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml_path', type=str, required=True, help='Path to xml file or directory storing xml files')
    ap.add_argument('--output_path', type=str, required=True, help='Directory to store output')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # load config
    label_list = {'face': 0, 'license-plate': 1}

    vocdataset_to_yolov1(args.xml_path, args.output_path, label_list)