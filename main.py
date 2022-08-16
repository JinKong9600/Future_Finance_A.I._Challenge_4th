from model import *
from utils import *

import sys
import argparse

# Argument and global variables
parser = argparse.ArgumentParser('Interface #should added')
parser.add_argument('--directory', type=str, default='./dataset/personal_datasets', help='dataset path')

parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')

try:
    args = parser.parse_args()

except:
    parser.print_help()
    sys.exit(0)

DIRECTORY = args.directory
LR = args.lr
DROP_OUT = args.drop_out

if __name__ == '__main__':
    images, labels = load_dataset(DIRECTORY)
    print(f'Data Loaded')
    model = Spatial_Information_Extraction_Network()
    for img_p, label in zip(images, labels):
        print(img_p, label)
        img = transform_img(img_p)
        print(img.shape)
        output = model(img)
        print(output.shape)
