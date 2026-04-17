"""
TFC Quantized Neural Network — MIMO/BPSK Training Script
=========================================================

This version replaces MNIST with a synthetic MIMO detection dataset generated from

    X = H Y + N

Assumptions used to keep the original FINN/Brevitas classification pipeline:
- Real-valued channel and noise.
- BPSK symbols, so each entry of Y is in {-1, +1}.
- The full transmitted vector Y is encoded as a single class index.
  Therefore, NUM_CLASSES = 2 ** N_rx.
- By default the network input is [X, vec(H)], because if H changes from sample
  to sample, the detector cannot recover Y from X alone.

This keeps the official TFC/SqrHingeLoss training structure but swaps the data
source and the input dimensionality.
"""

import os
import math
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from functools import reduce
from operator import mul
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

from brevitas.nn import QuantIdentity, QuantLinear
from brevitas_examples.bnn_pynq.models.common import CommonWeightQuant, CommonActQuant


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# --- Model architecture ---
IN_FEATURES      = (1,)           # overwritten dynamically once the dataset is built
IN_CHANNELS      = 1
OUT_FEATURES     = [64, 64, 64]
NUM_CLASSES      = None           # for this task, it will default to 2 ** N_rx

# --- Quantization ---
WEIGHT_BIT_WIDTH = 2
ACT_BIT_WIDTH    = 2
IN_BIT_WIDTH     = 2

# --- Training hyperparameters ---
EPOCHS           = 500
BATCH_SIZE       = 100
LEARNING_RATE    = 0.02
WEIGHT_DECAY     = 0
DROPOUT          = 0.2

# --- Synthetic MIMO dataset defaults ---
N_TX             = 4
N_RX             = 4
TRAIN_SAMPLES    = 60000
TEST_SAMPLES     = 10000
SNR_DB           = 10.0
INCLUDE_CHANNEL  = True
FIXED_CHANNEL    = False

# --- Misc ---
NUM_WORKERS      = 4
RANDOM_SEED      = 1
DATASET          = 'MIMO_BPSK'


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SQUARED HINGE LOSS
# ─────────────────────────────────────────────────────────────────────────────

class squared_hinge_loss(Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        output = 1. - predictions.mul(targets)
        output[output.le(0.)] = 0.
        loss = torch.mean(output.mul(output))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        output = 1. - predictions.mul(targets)
        output[output.le(0.)] = 0.
        grad_output.resize_as_(predictions).copy_(targets).mul_(-2.).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(predictions.numel())
        return grad_output, None


class SqrHingeLoss(nn.Module):
    def __init__(self):
        super(SqrHingeLoss, self).__init__()

    def forward(self, input, target):
        return squared_hinge_loss.apply(input, target)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TENSOR NORM
# ─────────────────────────────────────────────────────────────────────────────

class TensorNorm(nn.Module):
    def __init__(self, eps=1e-4, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            mean = x.mean()
            unbias_var = x.var(unbiased=True)
            biased_var = x.var(unbiased=False)
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
            )
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            return (x - mean) * inv_std * self.weight + self.bias
        else:
            return (
                (x - self.running_mean) / (self.running_var + self.eps).pow(0.5)
            ) * self.weight + self.bias


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class TFC(nn.Module):
    def __init__(
        self,
        num_classes,
        weight_bit_width=WEIGHT_BIT_WIDTH,
        act_bit_width=ACT_BIT_WIDTH,
        in_bit_width=IN_BIT_WIDTH,
        in_channels=IN_CHANNELS,
        out_features=None,
        in_features=IN_FEATURES,
    ):
        super(TFC, self).__init__()

        if out_features is None:
            out_features = OUT_FEATURES

        self.features = nn.ModuleList()

        self.features.append(
            QuantIdentity(act_quant=CommonActQuant, bit_width=in_bit_width)
        )
        self.features.append(nn.Dropout(p=DROPOUT))

        in_feat = reduce(mul, in_features)

        for out_feat in out_features:
            self.features.append(
                QuantLinear(
                    in_features=in_feat,
                    out_features=out_feat,
                    bias=False,
                    weight_bit_width=weight_bit_width,
                    weight_quant=CommonWeightQuant,
                )
            )
            in_feat = out_feat
            self.features.append(nn.BatchNorm1d(num_features=in_feat))
            self.features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )
            self.features.append(nn.Dropout(p=DROPOUT))

        self.features.append(
            QuantLinear(
                in_features=in_feat,
                out_features=num_classes,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant,
            )
        )
        self.features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # Keep the same preprocessing convention as the original FC model:
        # the dataset is normalized to [0, 1], then remapped here to [-1, 1].
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.features:
            x = mod(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SYNTHETIC MIMO DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def class_index_to_bpsk(labels, n_rx):
    """
    Convert integer class labels to BPSK symbol vectors in {-1, +1}.

    Example for n_rx = 4:
        class 0  -> [-1, -1, -1, -1]
        class 15 -> [+1, +1, +1, +1]

    The first output column corresponds to the most significant bit.
    """
    bit_positions = torch.arange(n_rx - 1, -1, -1, dtype=torch.long)
    bits = ((labels.unsqueeze(1) >> bit_positions) & 1).float()
    return 2.0 * bits - 1.0



def generate_mimo_samples(
    num_samples,
    n_tx,
    n_rx,
    snr_db,
    include_channel=True,
    fixed_h=None,
    seed=None,
):
    """
    Generate a real-valued MIMO/BPSK dataset according to

        X = H Y + N

    Shapes:
        Y : [num_samples, n_rx]
        H : [num_samples, n_tx, n_rx]
        X : [num_samples, n_tx]

    Labels are integer class IDs in [0, 2**n_rx - 1], each encoding one BPSK
    vector Y. By default the network input is [X, vec(H)].
    """
    if n_rx <= 0 or n_tx <= 0:
        raise ValueError('n_tx and n_rx must be positive integers')

    num_classes = 2 ** n_rx
    if num_classes < 2:
        raise ValueError('Need at least one transmit symbol to define classes')

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    y = class_index_to_bpsk(labels, n_rx)

    if fixed_h is None:
        # Normalized real Gaussian channel. With var = 1/n_rx, the clean signal
        # power stays around 1 when Y is BPSK.
        h = torch.randn(num_samples, n_tx, n_rx, generator=generator) / math.sqrt(n_rx)
    else:
        if fixed_h.shape != (n_tx, n_rx):
            raise ValueError(f'fixed_h must have shape {(n_tx, n_rx)}, got {tuple(fixed_h.shape)}')
        h = fixed_h.unsqueeze(0).repeat(num_samples, 1, 1)

    x_clean = torch.bmm(h, y.unsqueeze(-1)).squeeze(-1)

    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power = x_clean.pow(2).mean(dim=1, keepdim=True)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power + 1e-12)
    noise = noise_std * torch.randn(num_samples, n_tx, generator=generator)
    x = x_clean + noise

    if include_channel:
        features = torch.cat([x, h.reshape(num_samples, -1)], dim=1)
    else:
        features = x

    return features.float(), labels.long()



def normalize_to_unit_interval(train_x, test_x):
    """
    Normalize features to [0, 1] using a single scalar derived from the train set.
    This preserves the original TFC forward preprocessing x -> 2x - 1.
    """
    scale = train_x.abs().max()
    if not torch.isfinite(scale) or scale.item() == 0.0:
        scale = torch.tensor(1.0)

    train_x = torch.clamp(train_x / (2.0 * scale) + 0.5, 0.0, 1.0)
    test_x = torch.clamp(test_x / (2.0 * scale) + 0.5, 0.0, 1.0)
    return train_x, test_x, float(scale.item())



def get_dataloaders(
    datadir,
    batch_size,
    num_workers,
    n_tx,
    n_rx,
    train_samples,
    test_samples,
    snr_db,
    include_channel,
    fixed_channel,
    seed,
):
    """
    Build train/test DataLoaders for the synthetic MIMO problem.

    Important note:
    - If fixed_channel=False, include_channel should stay True; otherwise the
      mapping X -> Y is ambiguous because H changes but is not provided.
    """
    if (not include_channel) and (not fixed_channel):
        raise ValueError(
            'Ambiguous setup: H changes per sample but include_channel=False. '
            'Use include_channel=True or fixed_channel=True.'
        )

    fixed_h = None
    if fixed_channel:
        base_generator = torch.Generator().manual_seed(seed)
        fixed_h = torch.randn(n_tx, n_rx, generator=base_generator) / math.sqrt(n_rx)

    train_x, train_y = generate_mimo_samples(
        num_samples=train_samples,
        n_tx=n_tx,
        n_rx=n_rx,
        snr_db=snr_db,
        include_channel=include_channel,
        fixed_h=fixed_h,
        seed=seed,
    )
    test_x, test_y = generate_mimo_samples(
        num_samples=test_samples,
        n_tx=n_tx,
        n_rx=n_rx,
        snr_db=snr_db,
        include_channel=include_channel,
        fixed_h=fixed_h,
        seed=seed + 1,
    )

    train_x, test_x, input_scale = normalize_to_unit_interval(train_x, test_x)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    input_size = train_x.shape[1]
    dataset_meta = {
        'dataset_name': DATASET,
        'n_tx': n_tx,
        'n_rx': n_rx,
        'num_classes': 2 ** n_rx,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'snr_db': float(snr_db),
        'include_channel': bool(include_channel),
        'fixed_channel': bool(fixed_channel),
        'input_size': int(input_size),
        'input_scale': float(input_scale),
        'label_encoding': 'BPSK vector Y encoded as integer class index',
        'feature_layout': (
            '[X, vec(H)]' if include_channel else '[X]'
        ),
        'fixed_h': fixed_h,
    }

    return train_loader, test_loader, dataset_meta


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    criterion.train()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        target_onehot = torch.Tensor(target.size(0), num_classes).to(device, non_blocking=True)
        target_onehot.fill_(-1)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        output = model(data)
        loss = criterion(output, target_onehot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(model, 'clip_weights'):
            model.clip_weights(-1, 1)

        total_loss += loss.item()
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), 100.0 * correct / total



def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    criterion.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            target_onehot = torch.Tensor(target.size(0), num_classes).to(device, non_blocking=True)
            target_onehot.fill_(-1)
            target_onehot.scatter_(1, target.unsqueeze(1), 1)

            output = model(data)
            loss = criterion(output, target_onehot)

            total_loss += loss.item()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), 100.0 * correct / total



def save_checkpoint(model, optimizer, epoch, accuracy, output_dir, dataset_meta, is_best=False):
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_acc': accuracy,
        'dataset_meta': dataset_meta,
    }

    torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.tar'))

    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'best.tar'))
        print(f'  --> Best model saved at epoch {epoch} with accuracy {accuracy:.2f}%')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description='TFC Custom MIMO Training Script')
    parser.add_argument('--num_classes',      type=int,   default=None,
                        help='Defaults to 2 ** n_rx for BPSK vector classification')
    parser.add_argument('--weight_bit_width', type=int,   default=WEIGHT_BIT_WIDTH)
    parser.add_argument('--act_bit_width',    type=int,   default=ACT_BIT_WIDTH)
    parser.add_argument('--in_bit_width',     type=int,   default=IN_BIT_WIDTH)
    parser.add_argument('--epochs',           type=int,   default=EPOCHS)
    parser.add_argument('--batch_size',       type=int,   default=BATCH_SIZE)
    parser.add_argument('--lr',               type=float, default=LEARNING_RATE)
    parser.add_argument('--output_dir',       type=str,
                        default='/tmp/finn_dev_icaro/experiments/TFC_custom_mimo')
    parser.add_argument('--datadir',          type=str,   default='/tmp/mimo_data',
                        help='Unused for generated data, kept for compatibility')
    parser.add_argument('--resume',           type=str,   default=None,
                        help='Path to checkpoint to resume training from')

    parser.add_argument('--n_tx',             type=int,   default=N_TX)
    parser.add_argument('--n_rx',             type=int,   default=N_RX)
    parser.add_argument('--train_samples',    type=int,   default=TRAIN_SAMPLES)
    parser.add_argument('--test_samples',     type=int,   default=TEST_SAMPLES)
    parser.add_argument('--snr_db',           type=float, default=SNR_DB)

    parser.add_argument('--include_channel', dest='include_channel', action='store_true',
                        help='Use [X, vec(H)] as the network input (recommended)')
    parser.add_argument('--x_only', dest='include_channel', action='store_false',
                        help='Use only X as network input')
    parser.set_defaults(include_channel=INCLUDE_CHANNEL)

    parser.add_argument('--fixed_channel', action='store_true', default=FIXED_CHANNEL,
                        help='Use the same H for every sample')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    if args.num_classes is None:
        args.num_classes = 2 ** args.n_rx

    expected_num_classes = 2 ** args.n_rx
    if args.num_classes != expected_num_classes:
        raise ValueError(
            f'For BPSK vector classification, num_classes must be 2**n_rx = '
            f'{expected_num_classes}, got {args.num_classes}.'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cpu':
        print(f'WARNING: No GPU found. Estimated time: ~{args.epochs * 11 // 60} minutes')

    print(f'\nLoading {DATASET} dataset...')
    train_loader, test_loader, dataset_meta = get_dataloaders(
        datadir=args.datadir,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        n_tx=args.n_tx,
        n_rx=args.n_rx,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        snr_db=args.snr_db,
        include_channel=args.include_channel,
        fixed_channel=args.fixed_channel,
        seed=RANDOM_SEED,
    )
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Test samples:  {len(test_loader.dataset)}')
    print(f'Input size:    {dataset_meta["input_size"]}')
    print(f'Feature layout:{dataset_meta["feature_layout"]}')

    print(f'\nBuilding TFC model:')
    print(f'  In features:    ({dataset_meta["input_size"]},)')
    print(f'  Hidden layers:  {OUT_FEATURES}')
    print(f'  Output classes: {args.num_classes}')
    print(f'  Weight bits:    {args.weight_bit_width}')
    print(f'  Act bits:       {args.act_bit_width}')
    print(f'  Input bits:     {args.in_bit_width}')

    model = TFC(
        num_classes=args.num_classes,
        weight_bit_width=args.weight_bit_width,
        act_bit_width=args.act_bit_width,
        in_bit_width=args.in_bit_width,
        in_features=(dataset_meta['input_size'],),
    ).to(device)

    criterion = SqrHingeLoss().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    starting_epoch = 1
    best_acc = 0.0
    if args.resume is not None:
        print(f'\nResuming from checkpoint: {args.resume}')
        package = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(package['state_dict'])
        optimizer.load_state_dict(package['optim_dict'])
        starting_epoch = package.get('epoch', 1) + 1
        best_acc = package.get('best_val_acc', 0.0)
        model = model.to(device)
        print(f'Resumed from epoch {starting_epoch - 1}, best acc: {best_acc:.2f}%')

    print(f'\nStarting training for {args.epochs} epochs...')
    print(f'Loss:        SqrHingeLoss')
    print(f'Optimizer:   Adam lr={args.lr}')
    print(f'Scheduler:   FIXED — LR halved every 40 epochs')
    print(f'Clip:        QuantLinear weights clamped to [-1,1] each step')
    print(f'Dataset:     {DATASET}')
    print(f'N_tx/N_rx:   {args.n_tx}/{args.n_rx}')
    print(f'SNR(dB):     {args.snr_db}')
    print(f'Checkpoints: {args.output_dir}')
    print('-' * 75)

    for epoch in range(starting_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.num_classes
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, args.num_classes
        )

        if epoch % 40 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        current_lr = optimizer.param_groups[0]['lr']

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        save_checkpoint(
            model,
            optimizer,
            epoch,
            test_acc,
            args.output_dir,
            dataset_meta=dataset_meta,
            is_best=is_best,
        )

        print(
            f'Epoch {epoch:4d}/{args.epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Train Acc: {train_acc:.2f}% | '
            f'Test Loss: {test_loss:.4f} | '
            f'Test Acc: {test_acc:.2f}% | '
            f'Best: {best_acc:.2f}% | '
            f'LR: {current_lr:.6f}'
        )

    print('-' * 75)
    print(f'\nTraining complete.')
    print(f'Best test accuracy: {best_acc:.2f}%')
    print(f'Best model saved to: {os.path.join(args.output_dir, "best.tar")}')
    print(f'\nTo load this model in the FINN notebook:')
    print(f'''
import torch, sys
sys.path.insert(0, '/tmp/finn_dev_icaro')
from train_tfc_custom_mimo import TFC, TensorNorm

package = torch.load("{os.path.join(args.output_dir, 'best.tar')}", map_location='cpu')
meta = package["dataset_meta"]

model = TFC(
    num_classes=meta["num_classes"],
    weight_bit_width={args.weight_bit_width},
    act_bit_width={args.act_bit_width},
    in_bit_width={args.in_bit_width},
    in_features=(meta["input_size"],),
)
model.load_state_dict(package["state_dict"])
model.eval()
tfc = model
    ''')


if __name__ == '__main__':
    main()
