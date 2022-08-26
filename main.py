import sys
import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
import time
from model import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument and global variables
parser = argparse.ArgumentParser('Interface')

parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--num_fc', type=int, default=1, choices=[1, 2], help='# of fully-connected layer')
parser.add_argument('--fc_dim', type=int, default=32, help='fully-connected layer dimension')

parser.add_argument('--patch_size', type=int, default=32, help='patch size')
parser.add_argument('--ol_ratio', type=float, default=0.5, help='overlapping ratio')
parser.add_argument('--max_round', type=int, default=10, help='max round for early stopper')
parser.add_argument('--cal_method', type=str, default='all', choices=['max', 'mean', 'any', 'all'], help='calculating method in probability')
parser.add_argument('--th', type=float, default=0.5, help='threshold in calculating mask')

try:
    args = parser.parse_args()

except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_FC = 1
FC_DIM = None
PATCH_SIZE = args.patch_size
OL_RATIO = args.ol_ratio

NUM_EPOCH = args.n_epoch
MAX_ROUND = args.max_round
CAL_METHOD = args.cal_method
TH = args.th

test_list = pd.read_csv('test_list.csv', index_col=0)

MODEL_PERFORMANCE_PATH = f'./saved_models/model_perfomance_eval.csv'
try:
    saved_model = pd.read_csv(MODEL_PERFORMANCE_PATH, index_col=0)
    MODEL_NUM = saved_model.index[-1] + 1
except:
    MODEL_NUM = 0

MODEL_NAME = f'{MODEL_NUM}'.zfill(3)

MODEL_SAVE_PATH = f'./saved_models/{MODEL_NAME}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{MODEL_NAME}-{epoch}.pth'

spec = test_list.loc[MODEL_NUM]
BATCH_SIZE, LEARNING_RATE, NUM_FC, FC_DIM, PATCH_SIZE, OL_RATIO = spec[:]
BATCH_SIZE = int(BATCH_SIZE)
LEARNING_RATE = float(LEARNING_RATE)
NUM_FC = int(NUM_FC)
FC_DIM = int(FC_DIM)
PATCH_SIZE = int(PATCH_SIZE)
OL_RATIO = float(OL_RATIO)

print(f'MODEL_NUM : {MODEL_NAME}')
print(spec[:])

model = Model(num_fc=NUM_FC,
              fc_dim=FC_DIM
              )
model.to(device)
for param in model.CFEN.preprocessing.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(MAX_ROUND)

t_images, t_forgery_box = load_dataset('train')
# split train/val
num_data = len(t_images)
train_indices = np.arange(math.ceil(num_data * 0.8))
val_indices = np.arange(math.ceil(num_data * 0.8), num_data)

train_images, train_forgery_box = t_images[train_indices], t_forgery_box[train_indices]
val_images, val_forgery_box = t_images[val_indices], t_forgery_box[val_indices]
test_images, test_forgery_box = load_dataset('test')

def eval_one_epoch(model, images, forgery_box, data_saving):
    acc, ap, f1, auc, recall = [], [], [], [], []
    images_cut, forgery_box_cut = random_shuffle(images, forgery_box)

    with torch.no_grad():
        model.eval()

    for img_p, f_box in zip(images_cut, forgery_box_cut):
        narrow_img, patches, _, patch_indices_map, labels = \
            transform_img(img_p, f_box, PATCH_SIZE, OL_RATIO)

        discriminator = Discriminate(narrow_img.shape[1:], CAL_METHOD, TH)
        patch_indices = np.arange(patches.shape[0])
        num_instance = patches.shape[0]
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance, s_idx + BATCH_SIZE)

            patch_cut = patch_indices[s_idx:e_idx]
            target_patches = patches[patch_cut]
            mask_cut = patch_indices_map[patch_cut]

            pred_prob = model(target_patches).cpu().detach().numpy()
            discriminator.fill(mask_cut, pred_prob)

        output = discriminator.score().cpu().numpy()
        true_label = labels.cpu().numpy()

        assert output.shape == true_label.shape

        pred_label = output.ravel()
        true_label = true_label.ravel()

        acc.append((pred_label == true_label).mean())
        f1.append(f1_score(true_label, pred_label))
        recall.append(recall_score(true_label, pred_label))

    ACC = np.mean(acc)
    F1 = np.mean(f1)
    RECALL = np.mean(recall)

    if data_saving:
        new = [{'BATCH_SIZE': BATCH_SIZE,
                'LR': LEARNING_RATE,
                'NUM_FC': NUM_FC,
                'FC_DIM': FC_DIM,
                'PATCH_SIZE' : PATCH_SIZE,
                'OL_RATIO' : OL_RATIO,
                'ACCURACY': ACC,
                'F1_SCORE': F1,
                'RECALL_SCORE': RECALL}]
        new_model = pd.DataFrame.from_dict(new)
        try:
            saved_model = pd.read_csv(MODEL_PERFORMANCE_PATH, index_col=0)
            updated_model = saved_model.append(new_model)
            updated_model = updated_model.reset_index(drop=True)
            updated_model.to_csv(MODEL_PERFORMANCE_PATH)
        except:
            new_model.to_csv(MODEL_PERFORMANCE_PATH)

    return ACC, F1, RECALL


def run():
    print(f'Train : {len(train_images)} | Val : {len(val_images)} | Test : {len(test_images)}')

    for epoch in range(NUM_EPOCH):
        start = time.time()
        acc, ap, f1, auc, recall, m_loss = [], [], [], [], [], []
        images_cut, forgery_box_cut = random_shuffle(train_images, train_forgery_box)

        for img_p, f_box in zip(images_cut, forgery_box_cut):
            _, patches, labels, _, _ = transform_img(img_p, f_box, PATCH_SIZE, OL_RATIO)
            if not labels.sum():
                continue

            patch_indices = class_balancing(len(patches), labels, BATCH_SIZE)
            num_instance = len(patch_indices)
            num_batch = math.ceil(num_instance / BATCH_SIZE)

            for k in range(num_batch):
                s_idx = k * BATCH_SIZE
                e_idx = min(num_instance, s_idx + BATCH_SIZE)

                patch_cut = patch_indices[s_idx:e_idx]
                target_patches = patches[patch_cut]
                target_labels = labels[patch_cut]

                pred_prob = model(target_patches)
                loss = criterion(pred_prob, target_labels)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred_prob = pred_prob.cpu().detach().numpy()
                    pred_label = pred_prob > 0.5
                    true_label = target_labels.cpu().numpy()
                    acc.append((pred_label == true_label).mean())
                    ap.append(average_precision_score(true_label, pred_prob))
                    f1.append(f1_score(true_label, pred_label))
                    m_loss.append(loss.item())
                    auc.append(roc_auc_score(true_label, pred_prob))
                    recall.append(recall_score(true_label, pred_label))

        val_acc, val_f1, val_recall = eval_one_epoch(model, val_images, val_forgery_box, 0)
        print(f'MODEL NAME : {MODEL_NAME} | epoch : {epoch}, | '
              f'avg_time : {(time.time()-start)/(len(images_cut)*60):.3f}min')
        print(f'Epoch mean loss: {np.mean(m_loss)}')
        print(f'train acc: {np.mean(acc)}, val acc: {val_acc}')
        print(f'train f1: {np.mean(f1)}, val f1: {val_f1}')
        print(f'train recall: {np.mean(recall)}, val recall: {val_recall}')

        if early_stopper.early_stop_check(val_f1):
            print(f'No improvement over {early_stopper.max_round} epochs, stop training')
            best_epoch = early_stopper.best_epoch
            print(f'Loading the best model at epoch {best_epoch}')
            best_model_path = get_checkpoint_path(best_epoch)
            model.load_state_dict(torch.load(best_model_path))
            print(f'Loaded the best model at epoch {best_epoch} for inference')
            model.eval()
            os.remove(best_model_path)
            break
        else:
            if early_stopper.is_best:
                torch.save(model.state_dict(), get_checkpoint_path(epoch))
                print(f'Saved {MODEL_NAME}-{early_stopper.best_epoch}.pth')
                for i in range(epoch):
                    try:
                        os.remove(get_checkpoint_path(i))
                        print(f'Deleted {MODEL_NAME}-{i}.pth')
                    except:
                        continue

    test_acc, test_f1, test_recall = eval_one_epoch(model, test_images, test_forgery_box, 1)
    print(f'test acc: {test_acc}')
    print(f'test f1: {test_f1}')
    print(f'test recall: {test_recall}')
    print('Saving model')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('Models saved')

if __name__ == '__main__':
    run()
