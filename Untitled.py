import numpy as np
import torch.nn as nn
from PIL import Image
import cv2
import time
import sys
import argparse

# Argument and global variables
parser = argparse.ArgumentParser('Interface #should added')
parser.add_argument('--denoise_h', type=float, default=0.5, help='h variable for denoise function')
parser.add_argument('--th_method', type=str, default='otsu', choices=['otsu', 'adapted_gaussian', 'adapted_mean'], help='threshold method')
parser.add_argument('--pe_method', type=str, default='contours', choices=['connected components', 'contours'], help='patch extraction method')
parser.add_argument('--size_limit', type=bool, default=True, help='limit patch size')
parser.add_argument('--kernel_size', type=int, default=4, help='size of kernel')
parser.add_argument('--lower_limit', type=int, default=5, help='lower limit of patch pixels')
parser.add_argument('--higher_limit', type=int, default=100, help='higher limit of patch pixels')
parser.add_argument('--expand_ratio', type=float, default=1.2, help='expand ratio of extracted raw patch')

parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')

try:
    args = parser.parse_args()

except:
    parser.print_help()
    sys.exit(0)

DENOISE_H = args.denoise_h
TH_METHOD = args.th_method
PE_METHOD = args.pe_method
SIZE_LIMIT = args.size_limit
KERNEL_SIZE = args.kernel_size
LOWER_LIMIT = args.lower_limit
HIGHER_LIMIT = args.higher_limit
EXPAND_RATIO = args.expand_ratio

def imshow(target):
    Image.fromarray(target).show()


class Model(nn.Module):
    def __init__(self, raw_img_path):
        super(Model, self).__init__()
        self.raw_img = cv2.imread(raw_img_path)
        self.target_img = None
        self.blk_size = 9
        self.C = 5
        self.patch_l = np.zeros(4, dtype=np.int32)

    def preprocessing(self):
        # gray scaled
        self.target_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        # reduce noise
        self.target_img = cv2.fastNlMeansDenoising(self.target_img, None, 0.5)


    def thresholding(self):
        print(f'Start Thresholding | Method : {TH_METHOD}')
        if TH_METHOD == 'otsu':
            t, self.target_img = cv2.threshold(self.target_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            print('otsu threshold:', t)
        elif self.TH_METHOD == 'adapted_gaussian':
            self.target_img = cv2.adaptiveThreshold(self.target_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, self.blk_size, self.C)
        elif TH_METHOD == 'adapted_mean':
            self.target_img = cv2.adaptiveThreshold(self.target_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, self.blk_size, self.C)
        else:
            raise ValueError(f'Could not use threshold method Available : {args.th_method.choices}')

    def connected_components(self):
        (num_label, labels, stats, centroids) = \
            cv2.connectedComponentsWithStats(self.target_img, 8, cv2.CV_32S)
        print(stats.shape)
        return stats[:, :4]


    def contours(self):
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        result = cv2.morphologyEx(self.target_img, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(result, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def patch_extraction(self):
        print(f'Start Patch Extraction | Method : {PE_METHOD}')
        if PE_METHOD == 'connected components':
            patch_l = self.connected_components()
        elif PE_METHOD == 'contours':
            patch_l = self.contours()
        else:
            raise ValueError(f'Could not use patch extraction method | Available : {args.pe_method.choices}')

        if SIZE_LIMIT:
            for patch in patch_l:
                x, y, w, h = patch if PE_METHOD == 'connected components' else cv2.boundingRect(patch)

                if not (LOWER_LIMIT < w < HIGHER_LIMIT) and (LOWER_LIMIT < h < HIGHER_LIMIT):
                    continue
                self.patch_l = np.vstack((self.patch_l, [x, y, w, h]))
            self.patch_l = self.patch_l[1:]
        else:
            self.patch_l = patch_l

    # def boundary_expansion(self):


    def forward(self):
        print(self.raw_img.shape)
        self.preprocessing()
        self.thresholding()
        self.patch_extraction()
        print(self.patch_l.shape)

def run(test_img):
    test_img_path = f'./test_img/{test_img}'
    model = Model(test_img_path)
    model.forward()


test_img = '김진현_성적증명서.jpg'
run(test_img)
