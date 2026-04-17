import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay


DEFAULT_RANDOM_SEED = 1
DEFAULT_TEST_SEED_OFFSET = 1


# -----------------------------------------------------------------------------
# MIMO/BPSK dataset helpers
# -----------------------------------------------------------------------------

def _to_numpy_array(value: Any) -> Optional[np.ndarray]:
    """Convert tensors/lists/arrays to numpy arrays without importing torch eagerly."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def load_mimo_test_from_npz(npz_path: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load an exact saved MIMO test set exported during/after training."""
    data = np.load(npz_path, allow_pickle=False)

    for x_key in ("test_x", "x", "inputs"):
        if x_key in data:
            test_x = data[x_key]
            break
    else:
        raise KeyError(f"No test input array found in {npz_path}. Expected one of: test_x, x, inputs")

    for y_key in ("test_y", "y", "labels"):
        if y_key in data:
            test_y = data[y_key]
            break
    else:
        raise KeyError(f"No test label array found in {npz_path}. Expected one of: test_y, y, labels")

    meta: Dict[str, Any] = {}
    if "meta_json" in data:
        meta_raw = data["meta_json"]
        if isinstance(meta_raw, np.ndarray) and meta_raw.shape == ():
            meta_raw = meta_raw.item()
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        if isinstance(meta_raw, str):
            meta = json.loads(meta_raw)

    test_x = np.asarray(test_x, dtype=np.float32)
    test_y = np.asarray(test_y, dtype=np.int64)

    if max_samples is not None:
        test_x = test_x[:max_samples]
        test_y = test_y[:max_samples]

    return test_x, test_y, meta


# The generation path below is only used when the exact saved NPZ is unavailable.
# It recreates the dataset using the same torch RNG-based implementation as training.

def _class_index_to_bpsk(labels, n_rx):
    import torch  # type: ignore

    bit_positions = torch.arange(n_rx - 1, -1, -1, dtype=torch.long)
    bits = ((labels.unsqueeze(1) >> bit_positions) & 1).float()
    return 2.0 * bits - 1.0



def _generate_mimo_samples_with_torch(
    num_samples: int,
    n_tx: int,
    n_rx: int,
    snr_db: float,
    include_channel: bool = True,
    fixed_h=None,
    seed: Optional[int] = None,
):
    import math
    import torch  # type: ignore

    if n_tx <= 0 or n_rx <= 0:
        raise ValueError("n_tx and n_rx must be positive")

    num_classes = 2 ** n_rx
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    y = _class_index_to_bpsk(labels, n_rx)

    if fixed_h is None:
        h = torch.randn(num_samples, n_tx, n_rx, generator=generator) / math.sqrt(n_rx)
    else:
        if not isinstance(fixed_h, torch.Tensor):
            fixed_h = torch.as_tensor(fixed_h, dtype=torch.float32)
        if tuple(fixed_h.shape) != (n_tx, n_rx):
            raise ValueError(f"fixed_h must have shape {(n_tx, n_rx)}, got {tuple(fixed_h.shape)}")
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



def load_mimo_test_from_checkpoint(
    checkpoint_path: str,
    seed: Optional[int] = None,
    test_seed: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Regenerate the exact normalized MIMO test set from the training checkpoint.

    This path requires PyTorch on the machine running validation because the
    original dataset generator used torch RNG. Prefer --test_npz on lightweight
    board environments.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required to regenerate the MIMO test set from a checkpoint. "
            "Either install torch on the board or export the test set once as .npz "
            "and pass it with --test_npz."
        ) from exc

    package = torch.load(checkpoint_path, map_location="cpu")
    if "dataset_meta" not in package:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain dataset_meta")

    meta = dict(package["dataset_meta"])

    # Future-proof: use checkpoint-provided seeds when available; otherwise fall
    # back to the defaults used in the original training script.
    base_seed = seed
    if base_seed is None:
        base_seed = int(meta.get("seed", DEFAULT_RANDOM_SEED))

    eff_test_seed = test_seed
    if eff_test_seed is None:
        eff_test_seed = int(meta.get("test_seed", base_seed + DEFAULT_TEST_SEED_OFFSET))

    test_samples = int(meta["test_samples"])
    if max_samples is not None:
        test_samples = min(test_samples, int(max_samples))

    fixed_h = meta.get("fixed_h", None)
    if fixed_h is not None and not isinstance(fixed_h, torch.Tensor):
        fixed_h = torch.as_tensor(_to_numpy_array(fixed_h), dtype=torch.float32)

    test_x, test_y = _generate_mimo_samples_with_torch(
        num_samples=test_samples,
        n_tx=int(meta["n_tx"]),
        n_rx=int(meta["n_rx"]),
        snr_db=float(meta["snr_db"]),
        include_channel=bool(meta["include_channel"]),
        fixed_h=fixed_h,
        seed=eff_test_seed,
    )

    input_scale = meta.get("input_scale", None)
    if input_scale is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path} dataset_meta is missing input_scale, "
            "so the normalized board input cannot be reconstructed exactly."
        )

    input_scale = float(input_scale)
    test_x = torch.clamp(test_x / (2.0 * input_scale) + 0.5, 0.0, 1.0)

    # Make the metadata JSON-safe for optional logging/debugging.
    meta_out: Dict[str, Any] = {}
    for key, value in meta.items():
        if key == "fixed_h":
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            meta_out[key] = value
        else:
            try:
                meta_out[key] = _to_numpy_array(value).tolist()
            except Exception:
                meta_out[key] = str(value)

    meta_out["seed"] = base_seed
    meta_out["test_seed"] = eff_test_seed

    return (
        test_x.detach().cpu().numpy().astype(np.float32),
        test_y.detach().cpu().numpy().astype(np.int64),
        meta_out,
    )


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def load_dataset(args) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    dataset = args.dataset.lower()
    meta: Dict[str, Any] = {}

    if dataset == "mnist":
        from dataset_loading import mnist

        trainx, trainy, testx, testy, valx, valy = mnist.load_mnist_data(
            args.dataset_root, download=True, one_hot=False
        )
        return np.asarray(testx, dtype=np.float32), np.asarray(testy, dtype=np.int64), meta

    if dataset == "cifar10":
        from dataset_loading import cifar

        trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
            args.dataset_root, download=True, one_hot=False
        )
        return np.asarray(testx, dtype=np.float32), np.asarray(testy, dtype=np.int64), meta

    if dataset == "mimo_bpsk":
        if args.test_npz is not None:
            return load_mimo_test_from_npz(args.test_npz, max_samples=args.max_samples)
        if args.checkpoint is not None:
            return load_mimo_test_from_checkpoint(
                args.checkpoint,
                seed=args.seed,
                test_seed=args.test_seed,
                max_samples=args.max_samples,
            )
        raise ValueError(
            "For dataset=mimo_bpsk you must provide either --test_npz (preferred) "
            "or --checkpoint."
        )

    raise ValueError(f"Unrecognized dataset: {args.dataset}")



def validate(driver: FINNExampleOverlay, test_x: np.ndarray, test_y: np.ndarray, batchsize: int) -> float:
    test_x = np.asarray(test_x, dtype=np.float32)
    test_y = np.asarray(test_y, dtype=np.int64).reshape(-1)

    if test_x.shape[0] != test_y.shape[0]:
        raise ValueError(
            f"Feature/label size mismatch: {test_x.shape[0]} inputs vs {test_y.shape[0]} labels"
        )

    total = test_x.shape[0]
    ok = 0
    nok = 0

    for start in range(0, total, batchsize):
        end = min(start + batchsize, total)
        cur_x = test_x[start:end]
        cur_y = test_y[start:end]
        cur_bs = end - start

        if driver.batch_size != cur_bs:
            driver.batch_size = cur_bs

        try:
            cur_x = cur_x.reshape(driver.ishape_normal())
        except Exception as exc:
            raise ValueError(
                f"Unable to reshape input batch {cur_x.shape} to accelerator normal shape "
                f"{driver.ishape_normal()}."
            ) from exc

        pred = driver.execute(cur_x)
        pred = np.asarray(pred).reshape(-1)

        if pred.shape[0] != cur_y.shape[0]:
            raise ValueError(
                f"Output/label size mismatch in batch {start}:{end}: "
                f"pred {pred.shape} vs labels {cur_y.shape}"
            )

        batch_ok = int(np.sum(pred == cur_y))
        batch_nok = cur_bs - batch_ok
        ok += batch_ok
        nok += batch_nok
        batch_idx = start // batchsize + 1
        n_batches = (total + batchsize - 1) // batchsize
        print(f"batch {batch_idx} / {n_batches} : batch OK {batch_ok} NOK {batch_nok} | total OK {ok} NOK {nok}")

    return 100.0 * ok / total



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate top-1 accuracy for a FINN-generated accelerator"
    )
    parser.add_argument("--batchsize", help="number of samples for inference", type=int, default=100)
    parser.add_argument(
        "--dataset",
        help="dataset to use: mnist, cifar10, or mimo_bpsk",
        required=True,
    )
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma or alveo", default="zynq-iodma"
    )
    parser.add_argument(
        "--bitfile", help='name of bitfile (for example "resizer.bit")', default="resizer.bit"
    )
    parser.add_argument(
        "--dataset_root", help="dataset root dir for MNIST/CIFAR download/reuse", default="/tmp"
    )
    parser.add_argument(
        "--test_npz",
        help="Exact saved MIMO test set (.npz) with arrays test_x and test_y. Preferred for board validation.",
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        help="Training checkpoint (best.tar/checkpoint.tar) used to regenerate the MIMO test set exactly.",
        default=None,
    )
    parser.add_argument(
        "--seed",
        help="Override the base seed when regenerating MIMO data from a checkpoint. Defaults to the value in the checkpoint or 1.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--test_seed",
        help="Override the test seed when regenerating MIMO data from a checkpoint. Defaults to seed+1 unless stored in the checkpoint.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_samples",
        help="Optional cap on the number of validation samples.",
        type=int,
        default=None,
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    test_x, test_y, meta = load_dataset(args)

    total = test_x.shape[0]
    initial_batchsize = min(args.batchsize, total)
    if initial_batchsize <= 0:
        raise ValueError("No validation samples available")

    driver = FINNExampleOverlay(
        bitfile_name=args.bitfile,
        platform=args.platform,
        io_shape_dict=io_shape_dict,
        batch_size=initial_batchsize,
        runtime_weight_dir="runtime_weights/",
    )

    print(f"Loaded dataset: {args.dataset}")
    print(f"Validation samples: {total}")
    print(f"Input shape per sample: {tuple(test_x.shape[1:])}")
    print(f"Accelerator normal input shape: {driver.ishape_normal()}")
    if meta:
        short_meta = {k: v for k, v in meta.items() if k not in {"fixed_h"}}
        print("Dataset metadata:")
        print(json.dumps(short_meta, indent=2, sort_keys=True))

    acc = validate(driver, test_x, test_y, args.batchsize)
    print(f"Final accuracy: {acc:.6f}")
