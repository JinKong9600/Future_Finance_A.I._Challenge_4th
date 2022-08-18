import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import os
import xml.etree.ElementTree as ET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_shuffle(a, b, c):
    indices = np.arange(len(a))
    np.random.shuffle(indices)
    print(len(a))
    print(len(b))
    return a[indices], b[indices], c[indices]

def load_dataset(mode):
    paths = []
    forgery_box = []
    labels = []

    if mode == 'train':
        img_ = lambda x: f'./dataset/findit/train/img/{str(x).zfill(3)}.jpg'
        xml_ = lambda x: f'./dataset/findit/train/xml/{str(x).zfill(3)}.xml'


        for file_num in range(180):
            bbx = []
            img_path = img_(file_num)
            xml_path = xml_(file_num)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for forgeries in root:
                h, w, x, y = map(int, forgeries.attrib.values())
                bbx.append([x, y, h, w])
            paths.append(img_path)
            forgery_box.append(bbx)

        assert len(paths) == len(forgery_box)
        labels = [0]*len(paths)

    elif mode == 'val':
        img_ = lambda x: f'./dataset/findit/val/img/{x}.jpg'
        xml_path = f'./dataset/findit/val/labels.xml'

        tree = ET.parse(xml_path)
        root = tree.getroot()
        for f in root:
            id, label = f.attrib.get('id'), f.attrib.get('modified')
            img_path = img_(id)
            paths.append(img_path)
            labels.append(label)
        assert len(paths) == len(labels)
        forgery_box = [0]*len(paths)

    elif mode == 'test':
        img_ = lambda x: f'./dataset/personal_datasets/img/{x}.jpg'
        xml_path = f'./dataset/personal_datasets/labels.xml'

        tree = ET.parse(xml_path)
        root = tree.getroot()
        for f in root:
            id, label = f.attrib.get('id'), f.attrib.get('modified')
            img_path = img_(id)
            paths.append(img_path)
            labels.append(label)
        assert len(paths) == len(labels)
    else:
        raise ValueError('Check datasets')

    return np.array(paths), np.array(forgery_box, dtype=object), np.array(labels)


def transform_img(path, f_box, patch_size, overlapping_ratio):
    img = Image.open(path)
    img = F.to_tensor(img)

    (h, w) = img.shape[1:]
    kernel_size = np.lcm(8, patch_size)
    print(f'ADJUSTED_PATCH_SIZE : {kernel_size} X {kernel_size}')
    h_cut, w_cut = (h // kernel_size) * kernel_size, (w // kernel_size) * kernel_size
    upper_h_cut, left_w_cut = round((h-h_cut)/2), round((w-w_cut)/2)
    img = torch.narrow(img, 1, upper_h_cut, h_cut)
    img = torch.narrow(img, 2, left_w_cut, w_cut)
    img = img.unsqueeze(0)

    patches = img.unfold(2, kernel_size, int(patch_size*overlapping_ratio))\
        .unfold(3, kernel_size, int(kernel_size*overlapping_ratio))

    overlapping_patches = torch.empty((patches.shape[2]*patches.shape[3], 3, kernel_size, kernel_size))
    c = 0
    for i in range(patches.shape[2]):
        for j in range(patches.shape[3]):
            overlapping_patches[c] = patches[:, :, i, j, :]
            c += 1
    patch_labels = torch.zeros((patches.shape[2]*patches.shape[3]))
    print(overlapping_patches.shape)
    print(patch_labels.shape)
    exit()
    return overlapping_patches.to(device)

class EarlyStopMonitor(object):
    def __init__(self, max_round, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = -1
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.is_best = False

    def early_stop_check(self, curr_val):
        self.epoch_count += 1

        if not self.higher_better:
            curr_val *= -1

        if self.last_best is None:
            self.last_best = curr_val
            self.best_epoch = self.epoch_count
            self.is_best = True

        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            self.is_best = True

        else:
            self.num_round += 1
            self.is_best = False

        return self.num_round >= self.max_round

