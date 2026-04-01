import re

import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat

from hs_dsm_dataset import *
from models import resnet18, resnet18_adr
from models import co_cnn
from models import mft
from models import mvit
# from models import mamba
from models import dsymfuser
from models import mamba


from losses import lossfunction

import numpy as np
import time
import os
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cluster import KMeans
import torch.nn.functional as F
from models import dahgmn
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser("HSI")
A = parser.add_argument('--Weight_decay', type=float, default=5e-4, help='weight_decay')
parser.add_argument('--dataset', default='Houston', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'test'], default='train', help='testing mark')
parser.add_argument('--gpu', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=5, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patch', type=int, default=13, help='number of patches')
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--norm_type', type=str, choices=['pixelwise', 'bandwise'], default='bandwise', help='norm_type')
parser.add_argument('--select_type', type=str, choices=['normal', 'random'], default='normal', help='select_type')
parser.add_argument('--warmup_epoch', type=float, default=10, help='weight_decay')
parser.add_argument('--scheduler', type=str, default='cosine', help='scheduler')
parser.add_argument('--warmup_flag', type=bool, default=True, help='weight_decay')
parser.add_argument('--number_works', type=int, default=1, help='number_work')
parser.add_argument('--loss', type=str, default='criterion', help='losses fusion')
parser.add_argument('--model', type=str, default='resnet', help='model type')
parser.add_argument('--lam', type=float, default=0.1, help='lambda in distiall')
parser.add_argument('--sigma_scale', type=float, default=1.0,
                    help='scaling factor c for sigma = c * IQR(a), shared across all modalities and datasets')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# -------------------------------------------------------------------------------
def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


# -------------------------------------------------------------------------------
print("=================================== Parameters ===================================")
print_args(vars(args))
# -------------------------------------------------------------------------------
# Parameter Setting load seed
# -------------------------------------------------------------------------------
set_seed(args.seed)
# ---------------------------------------------------
# ----------------------------
# Dataloader
# -------------------------------------------------------------------------------
input_MultiModal, band_MultiModal, label_TR, label_TE = load_dataset(args.dataset, args.norm_type)
n_classes = np.max(label_TR)
print('n_classes is {}'.format(n_classes))
print('got the train_data_loader')

train_data_loader, val_data_loader = get_trian_val_loader(input_MultiModal, band_MultiModal, label_TR,
                                                          patch=13, select_type=args.select_type,
                                                          batch_size=args.batch_size,
                                                          num_workers=args.number_works,
                                                          distributed=False, rngsd1=args.seed)
print('got the test_data_loader')
test_data_loader = get_dataloader(input_MultiModal, band_MultiModal, label_TE,
                                  patch=13, select_type=args.select_type, batch_size=args.batch_size,
                                  num_workers=args.number_works,
                                  distributed=False, rngsd1=args.seed)


# -------------------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------------------
def copy_model_params(source_model, target_model):
    target_model.load_state_dict(source_model.state_dict())


if args.model == 'resnet':
    print('using resnet18')
    model = resnet18_adr.resnet(n_classes, band_MultiModal)
    model1 = resnet18_adr.resnet(n_classes, band_MultiModal)
    model2 = resnet18_adr.resnet(n_classes, band_MultiModal)
elif args.model == 'cnn':
    model = co_cnn.CO_CNN(band_MultiModal, n_classes)
    model1 = co_cnn.CO_CNN(band_MultiModal, n_classes)
    model2 = co_cnn.CO_CNN(band_MultiModal, n_classes)

elif args.model == 'mvit':
    model = mvit.MViT(
        patch_size=args.patch,
        num_patches=band_MultiModal,
        num_classes=n_classes,
        dim=64,
        depth=6,
        heads=4,
        mlp_dim=32,
        dropout=0.1,
        emb_dropout=0.1,
    )
    model1 = mvit.MViT(
        patch_size=args.patch,
        num_patches=band_MultiModal,
        num_classes=n_classes,
        dim=64,
        depth=6,
        heads=4,
        mlp_dim=32,
        dropout=0.1,
        emb_dropout=0.1,
    )
    model2 = mvit.MViT(
        patch_size=args.patch,
        num_patches=band_MultiModal,
        num_classes=n_classes,
        dim=64,
        depth=6,
        heads=4,
        mlp_dim=32,
        dropout=0.1,
        emb_dropout=0.1,
    )
elif args.model == 'mft':
    model = mft.MFT(
        FM=16,
        NC=band_MultiModal[0],
        NCLidar=band_MultiModal[1],
        Classes=n_classes,
        HSIOnly=False
    )
    model1 = mft.MFT(
        FM=16,
        NC=band_MultiModal[0],
        NCLidar=band_MultiModal[1],
        Classes=n_classes,
        HSIOnly=False
    )
    model2 = mft.MFT(
        FM=16,
        NC=band_MultiModal[0],
        NCLidar=band_MultiModal[1],
        Classes=n_classes,
        HSIOnly=False
    )
elif args.model == 'mamba':
    model = mamba.S2CrossMamba(
        num_classes=n_classes,
        AuHu=1,
        HS_c=band_MultiModal[0],
        Lidar_c=band_MultiModal[1]
    )
    model1 = mamba.S2CrossMamba(
        num_classes=n_classes,
        AuHu=1,
        HS_c=band_MultiModal[0],
        Lidar_c=band_MultiModal[1]
    )
    model2 = mamba.S2CrossMamba(
        num_classes=n_classes,
        AuHu=1,
        HS_c=band_MultiModal[0],
        Lidar_c=band_MultiModal[1]
    )
elif args.model == 'dsymfuser':
    model = dsymfuser.Proposed(
        hsi_dim=band_MultiModal[0],
        lidar_dim=band_MultiModal[1],
        num_classes=n_classes,
    )
    model1 = dsymfuser.Proposed(
        hsi_dim=band_MultiModal[0],
        lidar_dim=band_MultiModal[1],
        num_classes=n_classes,
    )
    model2 = dsymfuser.Proposed(
        hsi_dim=band_MultiModal[0],
        lidar_dim=band_MultiModal[1],
        num_classes=n_classes,
    )
else:
    print('check model name!')

model = model.cuda()


# -------------------------------------------------------------------------------
# Set losses
# -------------------------------------------------------------------------------

if  args.loss == 'criterion':
    loss_function = lossfunction.Criterion(n_classes)

loss_function = loss_function.cuda()


# ------------------------------------------------------------------------------
# Optimizer
# -------------------------------------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.Weight_decay)

# -------------------------------------------------------------------------------
# Scheduler
# -------------------------------------------------------------------------------
# cosine learning
epoch_iter = len(train_data_loader)
total_iter = epoch_iter * (args.epoch - args.warmup_epoch)
print('total_iter is :{}, every epoch has {} batch'.format(total_iter, epoch_iter))
if args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, verbose=False, eta_min=1e-18)
elif args.scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.gamma)
else:
    raise Exception('invalid scheduler {}'.format(args.scheduler))


def warm_up_lr(iterator, total_iterator, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = iterator * init_lr / total_iterator


# -------------------------------------------------------------------------------
# Train and test function
# -------------------------------------------------------------------------------

def load_modal(uni_modal, path):
    uni_modal = uni_modal.load_state_dict(torch.load(path))


def print_parameters(model):
    for name, param in model.named_parameters():
        print(name, param.mean().item(), param.std().item())


train_data_loader_all, _ = get_trian_val_loader(input_MultiModal, band_MultiModal, label_TR,
                                                patch=13, select_type=args.select_type, batch_size=1000,
                                                num_workers=args.number_works,
                                                distributed=False, rngsd1=args.seed)
modal1_avg = None
modal2_avg = None
for batch_idx, (batch_data, batch_target, idx) in enumerate(train_data_loader_all):
    if batch_idx == 0:
        modal1_avg = batch_data[:, 0:band_MultiModal[0], :, :]
        modal2_avg = torch.mean(batch_data[:, band_MultiModal[0]:band_MultiModal[0] + band_MultiModal[1],
                                :, :], dim=0, keepdim=True)
    else:
        modal1_avg = torch.cat([modal1_avg, batch_data[:, 0:band_MultiModal[0], :, :]], dim=0)
        modal2_avg = torch.cat([modal2_avg, batch_data[:, band_MultiModal[0]:band_MultiModal[0] + band_MultiModal[1],
                                            :, :]], dim=0)
with torch.no_grad():
    modal1_avg = torch.mean(modal1_avg, dim=0, keepdim=True).cuda()
    modal2_avg = torch.mean(modal2_avg, dim=0, keepdim=True).cuda()



class WarmupSigmaEstimator:

    def __init__(self, modal_names=('hs', 'lidar'), scale_c=1.0,
                 default_sigma=0.15, sigma_floor=0.02):

        self.modal_names = modal_names
        self.scale_c = scale_c
        self.default_sigma = default_sigma
        self.sigma_floor = sigma_floor


        self._buffer = {name: [] for name in modal_names}


        self._fixed_sigma = {name: None for name in modal_names}

        self.is_calibrated = False 

    def collect(self, modal_name, logits_uni):

        if self.is_calibrated:
            return

        with torch.no_grad():
            probs = F.softmax(logits_uni, dim=1)
            top2_vals, _ = torch.topk(probs, k=2, dim=1)
            sum_top2 = top2_vals.sum(dim=1) + 1e-8
            a = top2_vals[:, 0] / sum_top2 - 0.5   # a ∈ [0, 0.5]
            self._buffer[modal_name].append(a.cpu())

    def calibrate(self):
    
        print("\n" + "=" * 60)
        print("  [Sigma Calibration] sigma = c × (0.5 - median(a))")
        print(f"  [Scale] c = {self.scale_c}")
        print("=" * 60)

        for name in self.modal_names:
            if len(self._buffer[name]) == 0:
                print(f"    WARNING: No data for '{name}', fallback sigma={self.default_sigma}")
                self._fixed_sigma[name] = self.default_sigma
                continue

            all_a = torch.cat(self._buffer[name], dim=0).numpy()


            q25 = np.percentile(all_a, 25)
            q75 = np.percentile(all_a, 75)
            iqr = q75 - q25
            std_a = np.std(all_a)
            mean_a = np.mean(all_a)
            median_a = np.median(all_a)


            traversal_dist = 0.5 - median_a
            raw_sigma = self.scale_c * traversal_dist
            sigma = max(raw_sigma, self.sigma_floor)
            self._fixed_sigma[name] = sigma

            print(f"  [{name}] n_samples={len(all_a)}")
            print(f"    a_mean={mean_a:.4f}, a_median={median_a:.4f}, "
                  f"a_std={std_a:.4f}")
            print(f"    Q25={q25:.4f}, Q75={q75:.4f}, IQR={iqr:.4f}")
            print(f"    traversal_dist = 0.5 - {median_a:.4f} = {traversal_dist:.4f}")
            print(f"    => sigma = max({self.scale_c} × {traversal_dist:.4f}, "
                  f"{self.sigma_floor}) = {sigma:.4f}")

        self._buffer = {name: [] for name in self.modal_names}
        self.is_calibrated = True

        print("-" * 60)
        sigmas_str = ', '.join(f'{k}={v:.4f}' for k, v in self._fixed_sigma.items())
        print(f"  [Result] {sigmas_str}")
        print(f"  [Note] Larger sigma = more ambiguous modality = "
              f"wider window + gentler modulation")
        print("  [Status] Sigma locked. No further updates.")
        print("=" * 60 + "\n")

    def get_sigma(self, modal_name):

        if self.is_calibrated:
            return self._fixed_sigma[modal_name]
        else:
            return self.default_sigma

    def is_ready(self):
        return self.is_calibrated

sigma_estimator = WarmupSigmaEstimator(
    modal_names=('hs', 'lidar'), scale_c=args.sigma_scale,
    default_sigma=0.15, sigma_floor=0.02)

def compute_complementary_distillation_gaussian(logits_uni, logits_fusion, n_classes, sigma=0.2):
    probs_uni = F.softmax(logits_uni, dim=1)
    top2_vals, _ = torch.topk(probs_uni, k=2, dim=1)
    probs_fusion = F.softmax(logits_fusion, dim=1).detach()

    sum_top2 = top2_vals.sum(dim=1) + 1e-8
    a = top2_vals[:, 0] / sum_top2 - 0.5

    sigma_safe = max(float(sigma), 1e-4) if not isinstance(sigma, torch.Tensor) else torch.clamp(sigma, min=1e-4)


    weight = torch.exp(-(a ** 2) / (2 * sigma_safe ** 2))

    log_probs_uni = F.log_softmax(logits_uni, dim=1)
    kl_per_sample = F.kl_div(log_probs_uni, probs_fusion, reduction='none').sum(dim=1)
    kl_normalized = kl_per_sample / np.log(n_classes)


    loss = weight.mean() + (weight.detach() * kl_normalized).mean()


    info = {'weight_mean': weight.mean(), 'ambiguity': a, 'sigma': sigma}
    return loss, info


def train_epoch_MM(model, train_loader, optimizer, epoch, epoch_iter, warmup_epoch, sigma_est, puritys=None, **kwargs):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    for batch_idx, (batch_data, batch_target, idx) in enumerate(train_loader):
        # warmup
        if args.warmup_flag:
            if epoch < warmup_epoch:
                warm_up_lr(1 + epoch * epoch_iter + batch_idx, warmup_epoch * epoch_iter,
                           args.lr, optimizer)

        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        batch_pred, out_dict_m = model.forward(batch_data[:, 0:band_MultiModal[0], :, :],
                                               batch_data[:, band_MultiModal[0]:band_MultiModal[0] + band_MultiModal[1],
                                               :, :], batch_target, modal1_avg=modal1_avg, modal2_avg=modal2_avg,
                                               flag='train')

        out_dict = {
            'puritys': puritys
        }
        total_loss, loss_dict = loss_function(batch_pred, batch_target)

        loss_fr = 0.0

        if 'output1' in out_dict_m and 'output2' in out_dict_m:
            if out_dict_m['output1'] is not None and out_dict_m['output2'] is not None:


                if not sigma_est.is_ready():
                    sigma_est.collect('hs', out_dict_m['output1'])
                    sigma_est.collect('lidar', out_dict_m['output2'])


                if sigma_est.is_ready():
                    dis1, info1 = compute_complementary_distillation_gaussian(
                        out_dict_m['output1'], batch_pred, n_classes,
                        sigma=sigma_est.get_sigma('hs'))
                    dis2, info2 = compute_complementary_distillation_gaussian(
                        out_dict_m['output2'], batch_pred, n_classes,
                        sigma=sigma_est.get_sigma('lidar'))

                    loss_fr = (dis1 + dis2) * args.lam

        total_loss = total_loss + loss_fr
        total_loss.backward()

        # ----------------
        optimizer.step()

        # ----------------

        if args.warmup_flag:
            if epoch >= warmup_epoch:
                scheduler.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(total_loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


def valid_epoch_MM(model, valid_loader, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    for batch_idx, (batch_data, batch_target, idx) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_indices = idx.cuda()


        batch_pred, out_dict = model.forward(batch_data[:, 0:band_MultiModal[0], :, :],
                                             batch_data[:, band_MultiModal[0]:band_MultiModal[0] + band_MultiModal[1],
                                             :, :],
                                             batch_target,
                                             flag='eval')

        loss, _ = loss_function(batch_pred, batch_target, out_dict, None, model=model, batch_indices=batch_indices,
                             flag='eval')
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))

        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre, None


def test_epoch(model, test_loader):
    pre = np.array([])
    start_time = time.time()
    for batch_idx, (batch_data, batch_target, idx) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred, _ = model.forward(batch_data[:, 0:band_MultiModal[0], :, :],
                                      batch_data[:, band_MultiModal[0]:band_MultiModal[0] + band_MultiModal[1],
                                      :, :], batch_target,
                                      flag='eval')

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    end_time = time.time()
    print(f"Average inference latency: {end_time - start_time:.4f} ms")

    return pre


# -------------------------------------------------------------------------------
# Start
# -------------------------------------------------------------------------------
if args.flag_test == 'test':

    print("=================================== Testing ===================================")


    model.load_state_dict(torch.load(PATH))

    model.eval()

    pre_total = []

    test_acc, test_obj, tar_v, pre_v = valid_epoch_MM(model, test_data_loader, optimizer)
    OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
    print(">>> Inference finished!")

    print("=================================== Results ===================================")
    print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(OA_TE * 100, AA_TE * 100, Kappa_TE))
    np.set_printoptions(precision=2, suppress=True)
    print("CA: ", CA_TE * 100)


elif args.flag_test == 'train':

    folder_log = './exp/' + str(args.patch) + '/'
    exp_precision_dir = './exp/precision'
    folder_log_record = './log/' + str(args.patch) + '/' + str(args.dataset) + '/my/' + args.model + '/'
    if not os.path.exists(folder_log):
        os.makedirs(folder_log)
    if not os.path.exists(folder_log_record):
        os.makedirs(folder_log_record)
    if not os.path.exists(exp_precision_dir):
        os.makedirs(exp_precision_dir)
    state_log = os.path.join(folder_log_record, 'state_log.txt')
    f_state = open(state_log, 'w')

    best_checkpoint = {"OA_TE": 0.50}
    best_epoch = 0
    print("=================================== Training ===================================")
    tic = time.time()
    last_score_a = 0
    last_score_v = 0
    ratio = None

    for epoch in range(args.epoch):
        if args.warmup_flag:
            if epoch < args.warmup_epoch and epoch == 0: optimizer.param_groups[0]['lr'] = 0
            lr = optimizer.param_groups[0]['lr']
        model.train()
        loss_function.train()

        ratio = None
        puritys = None

        train_acc, train_obj, tar_t, pre_t = train_epoch_MM(model, train_data_loader, optimizer, epoch, epoch_iter,
                                                            args.warmup_epoch, sigma_estimator, puritys)

        if epoch == int(args.warmup_epoch) - 1 and not sigma_estimator.is_ready():
            sigma_estimator.calibrate()

        test_acc, test_obj, tar_v, pre_v, appendix = valid_epoch_MM(model, val_data_loader, optimizer)
        OA_TR, AA_TR, Kappa_TR, CA_TR = output_metric(tar_t, pre_t)
        if (epoch % args.test_freq == 0) | (epoch == args.epoch - 1):
            print('learning rate is {}'.format(optimizer.param_groups[0]['lr']))
            print("Epoch: {:03d} train_loss: {:.4f}, train_OA: {:.2f}".format(epoch + 1, train_obj, OA_TR * 100))

            model.eval()
            loss_function.eval()

            test_acc, test_obj, tar_v, pre_v, appendix = valid_epoch_MM(model, val_data_loader, optimizer)
            OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)

            print("Epoch: {:03d} test_loss: {:.4f}, test_OA: {:.2f}, test_AA: {:.2f}, test_Kappa: {:.4f}".format(
                epoch + 1, test_obj, OA_TE * 100, AA_TE * 100, Kappa_TE))


        PATH = folder_log + args.dataset +args.model+ str(epoch) + ' my' + '.pt'
        if OA_TE * 100 > best_checkpoint['OA_TE']:
            best_checkpoint = {'epoch': epoch, 'OA_TE': OA_TE * 100, 'AA_TE': AA_TE * 100, 'Kappa_TE': Kappa_TE,
                               'CA_TE': CA_TE * 100, 'pth': PATH}
            best_epoch = epoch
            PATH = folder_log + args.dataset + 'orgin' + ' ' + args.model + ' args.dataset ' + str(epoch) + '.pt'
            torch.save(model.state_dict(), PATH)


    toc = time.time()
    runtime = toc - tic
    print(">>> Training finished!")

    print(">>> Running time: {:.2f}".format(runtime))
    print("=================================== Results ===================================")
    print(">>> The peak performance in terms of OA is achieved at epoch", best_checkpoint['epoch'])
    print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(best_checkpoint['OA_TE'], best_checkpoint['AA_TE'],
                                                           best_checkpoint['Kappa_TE']))
    np.set_printoptions(precision=2, suppress=True)
    print("CA: ", best_checkpoint['CA_TE'])

    output_txt_path = os.path.join(folder_log, 'precision.txt')
    write_message = "Patch size {}, weight decay {}, learning rate {}, sigma_scale {}, sigma_hs {}, sigma_lidar {}, the best epoch {}, OA {}, AA {}, Kappa {}, run time {}".format(
        args.patch, args.Weight_decay, args.lr, args.sigma_scale,
        round(sigma_estimator.get_sigma('hs'), 4), round(sigma_estimator.get_sigma('lidar'), 4),
        best_checkpoint['epoch'],
        round(best_checkpoint['OA_TE'], 2), round(best_checkpoint['AA_TE'], 2), round(best_checkpoint['Kappa_TE'], 4),
        round(runtime, 2))

    output_txt_file = open(output_txt_path, "a")
    now = time.strftime("%c")
    output_txt_file.write(
        '=================================== Precision Log (%s) ===================================\n' % now)
    output_txt_file.write('%s\n' % write_message)
    output_txt_file.close()

    sigma_hs = round(sigma_estimator.get_sigma('hs'), 4)
    sigma_lidar = round(sigma_estimator.get_sigma('lidar'), 4)
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    precision_fname = f"{args.dataset}_{args.model}_sc{args.sigma_scale}.txt"
    precision_path = os.path.join(exp_precision_dir, precision_fname)

    log_lines = [
        f"--- lam={args.lam} | seed={args.seed} | {now} ---",
        f"  sigma_hs={sigma_hs}, sigma_lidar={sigma_lidar}",
        f"  best_epoch={best_checkpoint['epoch']}",
        f"  OA={round(best_checkpoint['OA_TE'], 2)}  "
        f"AA={round(best_checkpoint['AA_TE'], 2)}  "
        f"Kappa={round(best_checkpoint['Kappa_TE'], 4)}",
        f"  CA={np.array2string(best_checkpoint['CA_TE'], precision=2, separator=', ')}",
        f"  Runtime={round(runtime, 2)}s",
        "",
    ]

    if not os.path.exists(precision_path):
        header = [
            f"Dataset: {args.dataset}  |  Model: {args.model}  |  sigma_scale: {args.sigma_scale}",
            f"lr: {args.lr}  |  wd: {args.Weight_decay}  |  "
            f"warmup: {int(args.warmup_epoch)}  |  epochs: {args.epoch}  |  batch: {args.batch_size}",
            "=" * 70,
            "",
        ]
        with open(precision_path, "w") as f:
            f.write("\n".join(header) + "\n")

    with open(precision_path, "a") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\n  [Log] Results appended to: {precision_path}")