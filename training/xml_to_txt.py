import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_txt(path):
    xml_list = []

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'val_ann')
    txt_path = os.path.join(os.getcwd(), 'ground-truth')
    for xml_file in glob.glob(image_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            image_id = root.find('filename').text[:-4]  # file name
            width = int(root.find('size')[0].text)  # width
            height = int(root.find('size')[1].text)  # height
            obj_name = member[0].text    # class name
            left = int(member[4][0].text)    # xmin
            bottom = int(member[4][1].text)    # ymin
            right = int(member[4][2].text)    # xmax
            top = int(member[4][3].text)  # ymax

            print(image_id)
            print(obj_name)
            print(left)
            with open(txt_path + '\\' + image_id + ".txt", "a") as new_f:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, bottom, right, top))
    print("Conversion completed!")


main()
