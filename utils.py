import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import os
import xml.etree.ElementTree as ET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_shuffle(a, b):
    indices = np.arange(len(a))
    np.random.shuffle(indices)

    return a[indices], b[indices]

def class_balancing(p, l):
    l = l.cpu().numpy()
    if 0.5 < np.mean(l):
        major = np.where(l == 1)[0]
        minor = np.where(l == 0)[0]
    else:
        major = np.where(l == 0)[0]
        minor = np.where(l == 1)[0]
    assert len(major) + len(minor) == len(p)

    major = np.random.choice(major, len(minor), replace=False)

    return np.hstack((major, minor))

def load_dataset(mode):
    paths = []
    forgery_box = []

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
                h, w, y, x = map(int, forgeries.attrib.values())
                bbx.append([x, y, h, w])
            paths.append(img_path)
            forgery_box.append(bbx)

        assert len(paths) == len(forgery_box)

    # elif mode == 'val':
    #     img_ = lambda x: f'./dataset/findit/val/img/{x}.jpg'
    #     xml_path = f'./dataset/findit/val/labels.xml'
    #
    #     tree = ET.parse(xml_path)
    #     root = tree.getroot()
    #     for f in root:
    #         id, label = f.attrib.get('id'), f.attrib.get('modified')
    #         img_path = img_(id)
    #         paths.append(img_path)
    #         labels.append(label)
    #     assert len(paths) == len(labels)
    #     forgery_box = [0]*len(paths)

    # elif mode == 'test':
    #     img_ = lambda x: f'./dataset/personal_datasets/img/{x}.jpg'
    #     xml_path = f'./dataset/personal_datasets/labels.xml'
    #
    #     tree = ET.parse(xml_path)
    #     root = tree.getroot()
    #     for f in root:
    #         id, label = f.attrib.get('id'), f.attrib.get('modified')
    #         img_path = img_(id)
    #         paths.append(img_path)
    #         labels.append(label)
    #     assert len(paths) == len(labels)
    else:
        raise ValueError('Check datasets')

    return np.array(paths), np.array(forgery_box, dtype=object)


def transform_img(path, f_box, patch_size, overlapping_ratio):
    img = Image.open(path)
    img = F.to_tensor(img)

    (h, w) = img.shape[1:]
    kernel_size = int(np.lcm(8, patch_size))
    stride = int(kernel_size * overlapping_ratio)

    # print(f'ADJUSTED_PATCH_SIZE : {kernel_size} X {kernel_size}')
    h_cut, w_cut = (h // kernel_size) * kernel_size, (w // kernel_size) * kernel_size
    upper_h_cut, left_w_cut = round((h-h_cut)/2), round((w-w_cut)/2)

    img = torch.narrow(img, 1, upper_h_cut, h_cut)
    img = torch.narrow(img, 2, left_w_cut, w_cut)
    img = img.unsqueeze(0)

    f_box = list(map(lambda x: [x[0]-upper_h_cut,
                                x[1]-left_w_cut,
                                x[2],
                                x[3]], f_box))

    patches = img.unfold(2, kernel_size, int(patch_size*overlapping_ratio))\
        .unfold(3, kernel_size, int(kernel_size*overlapping_ratio))

    overlapping_patches = torch.empty((patches.shape[2]*patches.shape[3], 3, kernel_size, kernel_size))
    patch_labels_map = torch.zeros((img.shape[2], img.shape[3]))
    patch_labels = torch.zeros((patches.shape[2]*patches.shape[3]), dtype=torch.float)

    for x, y, h, w in f_box:
        patch_labels_map[x:x+h, y:y+w] = 1

    c = 0
    for i in range(patches.shape[2]):
        for j in range(patches.shape[3]):
            overlapping_patches[c] = patches[:, :, i, j, :]

            sliced_label_map = patch_labels_map[i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
            patch_labels[c] = sliced_label_map.max()
            c += 1

    return overlapping_patches.to(device), patch_labels.to(device)

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

