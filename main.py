import sys
import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score

from model import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument and global variables
parser = argparse.ArgumentParser('Interface')
parser.add_argument('--patch_size', type=int, default=32, help='patch size')
parser.add_argument('--ol_ratio', type=float, default=0.5, help='overlapping ratio')

parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--num_fc', type=int, default=1, choices=[1, 2], help='# of fully-connected layer')
parser.add_argument('--fc_dim', type=int, default=32, help='fully-connected layer dimension')
parser.add_argument('--max_round', type=int, default=10, help='max round for early stopper')

try:
    args = parser.parse_args()

except:
    parser.print_help()
    sys.exit(0)

PATCH_SIZE = args.patch_size
OL_RATIO = args.ol_ratio
NUM_EPOCH = args.n_epoch
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_FC = 1
FC_DIM = None
MAX_ROUND = args.max_round

MODEL_NAME = f'{BATCH_SIZE}-{LEARNING_RATE}-{NUM_FC}-{FC_DIM}-{PATCH_SIZE}-{OL_RATIO}'
MODEL_SAVE_PATH = f'./saved_models/{MODEL_NAME}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{MODEL_NAME}-{epoch}.pth'

MODEL_PERFORMANCE_PATH = f'./saved_models/model_perfomance_eval.csv'

model = Model(num_fc=NUM_FC,
              fc_dim=FC_DIM
              )
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(MAX_ROUND)


def eval_one_epoch(model, val_images, val_forgery_box, data_saving):
    acc, ap, f1, auc, recall = [], [], [], [], []
    id_l, patch_l = [], []
    images_cut, forgery_box_cut = random_shuffle(val_images, val_forgery_box)

    with torch.no_grad():
        model.eval()

    for img_p, f_box in zip(images_cut, forgery_box_cut):
        patches, labels = transform_img(img_p, f_box, PATCH_SIZE, OL_RATIO)
        if not labels.sum():
            continue

        patch_indices = class_balancing(patches, labels)
        num_instance = len(patch_indices)
        num_batch = math.ceil(num_instance / BATCH_SIZE)
        np.random.shuffle(patch_indices)

        id_l.append(img_p)
        patch_l.append(labels[patch_indices].cpu().numpy())

        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)

            patch_cut = patch_indices[s_idx:e_idx]
            target_patches = patches[patch_cut]
            target_labels = labels[patch_cut]

            pred_prob = model(target_patches).cpu().numpy()
            pred_label = pred_prob > 0.5
            true_label = target_labels

            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_prob))
            f1.append(f1_score(true_label, pred_label))
            # auc.append(roc_auc_score(true_label, pred_prob))
            recall.append(recall_score(true_label, pred_label))

    ACC = np.mean(acc)
    AP = np.mean(ap)
    F1 = np.mean(f1)
    # AUC = np.mean(auc)
    RECALL = np.mean(recall)


    if data_saving:
        new = [{'NUM_FC': NUM_FC,
                'FC_DIM': FC_DIM,
                'NUM_GRAPH': len(id_l),
                'ACCURACY': ACC,
                # 'AUC_ROC_SCORE': AUC,
                'AP_SCORE': AP,
                'RECALL_SCORE': RECALL,
                'F1_SCORE': F1}]
        new_model = pd.DataFrame.from_dict(new)
        try:
            saved_model = pd.read_csv(MODEL_PERFORMANCE_PATH, index_col=0)
            updated_model = saved_model.append(new_model)
            updated_model = updated_model.reset_index(drop=True)
            updated_model.to_csv(MODEL_PERFORMANCE_PATH)
        except:
            new_model.to_csv(MODEL_PERFORMANCE_PATH)


    return ACC, AP, F1, RECALL


def run():
    images, forgery_box = load_dataset('train')
    # split train/val
    num_data = len(images)

    train_indices = np.arange(math.ceil(num_data*0.8))
    val_indices = np.arange(math.ceil(num_data*0.8), num_data)

    train_images, train_forgery_box = images[train_indices], forgery_box[train_indices]
    val_images, val_forgery_box = images[val_indices], forgery_box[val_indices]

    print(f'Num Images : {num_data}')
    print(f'Train Images : {len(train_images)}')
    print(f'Val Images : {len(val_images)}')

    for epoch in range(NUM_EPOCH):
        acc, ap, f1, auc, recall, m_loss = [], [], [], [], [], []
        images_cut, forgery_box_cut = random_shuffle(train_images, train_forgery_box)

        for img_p, f_box in zip(images_cut, forgery_box_cut):
            patches, labels = transform_img(img_p, f_box, PATCH_SIZE, OL_RATIO)
            if not labels.sum():
                continue

            patch_indices = class_balancing(patches, labels)
            num_instance = len(patch_indices)
            num_batch = math.ceil(num_instance / BATCH_SIZE)
            np.random.shuffle(patch_indices)
            for k in range(num_batch):
                s_idx = k * BATCH_SIZE
                e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)

                patch_cut = patch_indices[s_idx:e_idx]
                target_patches = patches[patch_cut]
                target_labels = labels[patch_cut]

                pred_prob = model(target_patches)
                loss = criterion(pred_prob, target_labels)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred_prob = pred_prob.cpu().numpy()
                    pred_label = pred_prob > 0.5
                    true_label = target_labels.cpu().numpy()
                    acc.append((pred_label == true_label).mean())
                    ap.append(average_precision_score(true_label, pred_prob))
                    f1.append(f1_score(true_label, pred_label))
                    m_loss.append(loss.item())
                    # auc.append(roc_auc_score(true_label, pred_prob))
                    recall.append(recall_score(true_label, pred_label))

        val_acc, val_ap, val_f1, val_recall = eval_one_epoch(model, val_images, val_forgery_box, 1)
        print(f'epoch : {epoch}')
        print(f'Epoch mean loss: {np.mean(m_loss)}')
        print(f'train acc: {np.mean(acc)}, val acc: {val_acc}')
        # print(f'train auc: {np.mean(auc)}, val auc: {val_auc}')
        print(f'train ap: {np.mean(ap)}, val ap: {val_ap}')
        print(f'train recall: {np.mean(recall)}, val ap: {val_recall}')
        print(f'train f1: {np.mean(f1)}, val f1: {val_f1}')

        if early_stopper.early_stop_check(np.mean(m_loss)):
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

    print('Saving model')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('Models saved')

if __name__ == '__main__':
    run()
