import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
from hs_dsm_dataset import load_dataset, get_dataloader
from models import resnet18_adr
# -------------------------------------------------

# ================= config=================
DATASET_NAME = 'Augsburg'
BANDS = [180, 4]
N_CLASSES = 7
BATCH_SIZE = 64

NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

CKPT_BASELINE = "./checkpt/Augsburgorgin_ resnet args.dataset 190.pt"
CKPT_OURS = "./checkpt/Augsburgidm_ resnet args.dataset 60.pt"
SAVE_DIR = './paper_figures/full_corruption_physical'


os.makedirs(SAVE_DIR, exist_ok=True)


def add_full_physical_noise(tensor, level, mode_type='hs'):
    """
    tensor: [B,C,H,W]
    level: 0~1）
    """
    if level <= 0:
        return tensor

    std = tensor.std(dim=(2, 3), keepdim=True) + 1e-8

    if mode_type == 'hs':

        global_std = tensor.std() + 1e-8
        noise = torch.randn_like(tensor) * global_std * level
        return tensor + noise



    elif mode_type == 'sar':

        global_std = tensor.std() + 1e-8


        additive = torch.randn_like(tensor) * global_std * level


        spatial_noise = torch.randn(

            tensor.shape[0], 1,

            tensor.shape[2], tensor.shape[3],

            device=tensor.device

        ).expand_as(tensor) * global_std * level * 0.5

        return tensor + additive + spatial_noise

    elif mode_type == 'lidar':

        mask = (torch.rand_like(tensor) < 0.3).float()
        spike = torch.randn_like(tensor) * std * level * 5.0
        return tensor + mask * spike

    return tensor



@torch.no_grad()
def test_accuracy_full_corruption_pair(model_base, model_ours, loader, noise_level, attack_mode):

    model_base.eval()
    model_ours.eval()

    correct_b, correct_o, total = 0, 0, 0
    aux_type = 'sar' if DATASET_NAME == 'Augsburg' else 'lidar'

    for batch_data, batch_target, _ in loader:
        batch_data = batch_data.cuda(non_blocking=True)
        batch_target = batch_target.cuda(non_blocking=True)


        hs = batch_data[:, :BANDS[0], :, :].clone()
        aux = batch_data[:, BANDS[0]:BANDS[0] + BANDS[1], :, :].clone()


        hs_noisy = hs
        aux_noisy = aux

        if attack_mode in ['optical', 'both']:
            hs_noisy = add_full_physical_noise(hs_noisy, noise_level, mode_type='hs')
        if attack_mode in ['aux', 'both']:
            aux_noisy = add_full_physical_noise(aux_noisy, noise_level, mode_type=aux_type)

        out_b, _ = model_base(hs_noisy, aux_noisy, flag='eval')
        out_o, _ = model_ours(hs_noisy, aux_noisy, flag='eval')

        pred_b = out_b.argmax(dim=1)
        pred_o = out_o.argmax(dim=1)

        total += batch_target.size(0)
        correct_b += (pred_b == batch_target).sum().item()
        correct_o += (pred_o == batch_target).sum().item()

    acc_b = correct_b / max(total, 1) * 100.0
    acc_o = correct_o / max(total, 1) * 100.0
    return acc_b, acc_o


# ================= 绘图逻辑 =================
def plot_full_corruption_results(results, noise_levels):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

    titles = [
        '(a) Optical Corruption (Full HSI)',
        '(b) Auxiliary Corruption (Full SAR/LiDAR)',
        '(c) Joint Corruption (Full Both)'
    ]
    modes = ['optical', 'aux', 'both']

    for i, mode in enumerate(modes):
        ax = axes[i]
        res = results[mode]

        ax.plot(noise_levels, res['base'], 'o--', color='#7f7f7f', label='Baseline', lw=2.5)
        ax.plot(noise_levels, res['ours'], '*-', color='#d62728', label='Ours (ADR)', lw=2.5)

        # gap 可视化（ours-base）
        base_arr = np.array(res['base'])
        ours_arr = np.array(res['ours'])
        ax.fill_between(noise_levels, base_arr, ours_arr, color='#d62728', alpha=0.10)

        ax.set_title(titles[i], fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Relative Noise Level ($\\sigma$)', fontsize=14)

        if i == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=14)
            ax.legend(loc='lower left', fontsize=12)

        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)


        gaps = ours_arr - base_arr
        max_idx = int(np.argmax(gaps))
        if gaps[max_idx] > 0.5:
            ax.annotate(
                f'Gap: +{gaps[max_idx]:.1f}%',
                xy=(noise_levels[max_idx], (base_arr[max_idx] + ours_arr[max_idx]) / 2),
                xytext=(0, 30), textcoords='offset points', ha='center',
                color='#b30000', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#b30000')
            )

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f'Full_Corruption_{DATASET_NAME}.pdf')
    plt.savefig(out_path, dpi=300)
    plt.show()
    print("[OK] saved:", out_path)


def main():
    # ===== data =====
    input_MM, band_MM, _, label_TE = load_dataset(DATASET_NAME, 'bandwise')
    test_loader = get_dataloader(
        input_MM, band_MM, label_TE,
        batch_size=BATCH_SIZE,
        select_type='normal',
        num_workers=4,
        distributed=False,
        rngsd1=5,
        patch=13,
    )

    # ===== models =====
    model_base = resnet18_adr.resnet(N_CLASSES, BANDS).cuda()
    model_base.load_state_dict(torch.load(CKPT_BASELINE, map_location='cpu'), strict=False)

    model_ours = resnet18_adr.resnet(N_CLASSES, BANDS).cuda()
    model_ours.load_state_dict(torch.load(CKPT_OURS, map_location='cpu'), strict=False)

    results = {m: {'base': [], 'ours': []} for m in ['optical', 'aux', 'both']}

    for mode in ['optical', 'aux', 'both']:
        print(f"\n>>> Mode: {mode.upper()} (100% Corruption, SAME noise for both models)")
        for lvl in NOISE_LEVELS:
            acc_b, acc_o = test_accuracy_full_corruption_pair(
                model_base, model_ours, test_loader, lvl, mode
            )
            results[mode]['base'].append(acc_b)
            results[mode]['ours'].append(acc_o)
            print(f"  Level {lvl:.2f}: Base={acc_b:.2f}%, Ours={acc_o:.2f}%")

    plot_full_corruption_results(results, NOISE_LEVELS)


if __name__ == '__main__':
    main()
