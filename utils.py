import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import os
import math
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_shuffle(a, b):
    indices = np.arange(len(a))
    np.random.shuffle(indices)

    return a[indices], b[indices]

def class_balancing(len_p, l, batch):
    l = l.cpu().numpy()
    if 0.5 < np.mean(l):
        major = np.where(l == 1)[0]
        minor = np.where(l == 0)[0]
    else:
        major = np.where(l == 0)[0]
        minor = np.where(l == 1)[0]
    assert len(major) + len(minor) == len_p

    major = np.random.choice(major, len(minor), replace=False)
    patch_indices = np.hstack((major, minor))

    recur = True
    while recur:
        np.random.shuffle(patch_indices)
        temp = l[patch_indices]

        for k in range(math.ceil(len(temp) / batch)):
            s_idx = k * batch
            e_idx = min(len(temp), s_idx + batch)
            if not (0 < temp[s_idx:e_idx].mean() < 1):
                break
        else:
            recur = False
    return patch_indices

def load_dataset(data_type):
    paths = []
    forgery_box = []

    if data_type == 'train':
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

        return np.array(paths), np.array(forgery_box, dtype=object)

    elif data_type == 'test':
        path = f'./dataset/personal_datasets'
        paths = []

        pristine = [os.path.join(path, '0', f) for f in os.listdir(os.path.join(path, '0')) if
                    os.path.isfile(os.path.join(path, '0', f))]
        forged = [os.path.join(path, '1', f) for f in os.listdir(os.path.join(path, '1')) if
                  os.path.isfile(os.path.join(path, '1', f))]

        paths.extend(pristine)
        paths.extend(forged)

        labels = np.concatenate((np.zeros(len(pristine)), (np.ones(len(forged)))))

        return paths, labels
    else:
        raise ValueError('Check datasets')

def demo_transform(path, patch_size, overlapping_ratio):
    img = Image.open(path)
    img = F.to_tensor(img)
    img.to(device)
    raw_img = img.clone().detach()
    (h, w) = img.shape[1:]
    kernel_size = int(np.lcm(8, patch_size))
    stride = int(kernel_size * overlapping_ratio)

    h_cut, w_cut = (h // kernel_size) * kernel_size, (w // kernel_size) * kernel_size
    upper_h_cut, left_w_cut = round((h - h_cut) / 2), round((w - w_cut) / 2)

    img = torch.narrow(img, 1, upper_h_cut, h_cut)
    img = torch.narrow(img, 2, left_w_cut, w_cut)
    narrow_img = img.clone().detach()
    img = img.unsqueeze(0)

    patches = img.unfold(2, kernel_size, int(patch_size * overlapping_ratio)) \
        .unfold(3, kernel_size, int(kernel_size * overlapping_ratio))

    overlapping_patches = torch.empty((patches.shape[2] * patches.shape[3], 3, kernel_size, kernel_size), device=device)
    patch_indices_map = np.zeros((overlapping_patches.shape[0], 4), dtype=np.int32)
    c = 0
    for i in range(patches.shape[2]):
        for j in range(patches.shape[3]):
            overlapping_patches[c] = patches[:, :, i, j, :]
            patch_indices_map[c] = [i*stride, i*stride+kernel_size, j*stride, j*stride+kernel_size]
            c += 1

    return raw_img, narrow_img, overlapping_patches, patch_indices_map


def transform_img(path, f_box, patch_size, overlapping_ratio):
    img = Image.open(path)
    img = F.to_tensor(img)
    img.to(device)
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

    overlapping_patches = torch.empty((patches.shape[2]*patches.shape[3], 3, kernel_size, kernel_size), device=device)
    patch_labels_map = torch.zeros((img.shape[2], img.shape[3]), device=device)
    patch_labels = torch.zeros((patches.shape[2]*patches.shape[3]), dtype=torch.float, device=device)

    for x, y, h, w in f_box:
        patch_labels_map[x:x+h, y:y+w] = 1

    c = 0
    for i in range(patches.shape[2]):
        for j in range(patches.shape[3]):
            overlapping_patches[c] = patches[:, :, i, j, :]

            sliced_label_map = patch_labels_map[i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
            patch_labels[c] = sliced_label_map.max()
            c += 1

    return overlapping_patches, patch_labels

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

class Discriminate():
    def __init__(self, img_shape, visualize, cal_method, th):

        self.prob_map = torch.zeros(img_shape, device=device)
        self.cnt = torch.zeros(img_shape, device=device)
        self.visualize = visualize
        self.method = cal_method
        self.th = th

        self.mask = None
        self.output = None

    def fill(self, mask_cut, prob):
        for (x_s, x_d, y_s, y_d), p in zip(mask_cut, prob):
            if self.method == 'max':
                self.prob_map[0][x_s:x_d, y_s:y_d] = \
                    torch.maximum(self.prob_map[0][x_s:x_d, y_s:y_d],
                                  torch.ones((x_d-x_s, y_d-y_s), device=device) * p)

            elif self.method == 'mean':
                self.prob_map[0][x_s:x_d, y_s:y_d] += p
                self.cnt[0][x_s:x_d, y_s:y_d] += 1

    def score(self):
        if self.method == 'max':
            self.output = self.prob_map

        elif self.method == 'mean':
            self.output = torch.div(self.prob_map, self.cnt)

        self.mask = torch.as_tensor(self.output > self.th, dtype=torch.float32)
        return self.output, self.mask

def tf(k):
    return k.permute(1, 2, 0)

def visualize(raw, modi, prob, mask):
    raw = tf(raw)
    modi = tf(modi)
    prob = tf(prob)
    mask = tf(mask)

    plt.imshow(raw)
    plt.show()
    plt.imshow(modi)
    plt.show()

    fig, axs = plt.subplots(1, 4)
    a = axs[0].imshow(prob, vmin=0, vmax=1)
    plt.colorbar(a, ax=axs[1])

    a = axs[2].imshow(mask, vmin=0, vmax=1)
    plt.colorbar(a, ax=axs[4])
    plt.show()

