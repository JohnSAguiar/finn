"""
Export the exact normalized MIMO/BPSK test set from a training checkpoint.

This is the safest path for board validation because it avoids depending on
PyTorch RNG availability/version on the board. The board-side validator can then
load the resulting .npz with NumPy only.
"""

import argparse
import json
from typing import Any, Dict, Optional

import numpy as np


DEFAULT_RANDOM_SEED = 1
DEFAULT_TEST_SEED_OFFSET = 1



def _to_numpy_array(value: Any) -> Optional[np.ndarray]:
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
    include_channel: bool,
    fixed_h,
    seed: int,
):
    import math
    import torch  # type: ignore

    num_classes = 2 ** n_rx
    generator = torch.Generator().manual_seed(seed)

    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    y = _class_index_to_bpsk(labels, n_rx)

    if fixed_h is None:
        h = torch.randn(num_samples, n_tx, n_rx, generator=generator) / math.sqrt(n_rx)
    else:
        if not isinstance(fixed_h, torch.Tensor):
            fixed_h = torch.as_tensor(fixed_h, dtype=torch.float32)
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



def export_test_set(checkpoint_path: str, output_path: str, seed: Optional[int], test_seed: Optional[int], max_samples: Optional[int]) -> Dict[str, Any]:
    import torch  # type: ignore

    package = torch.load(checkpoint_path, map_location="cpu")
    if "dataset_meta" not in package:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain dataset_meta")

    meta = dict(package["dataset_meta"])
    base_seed = int(meta.get("seed", DEFAULT_RANDOM_SEED)) if seed is None else int(seed)
    eff_test_seed = int(meta.get("test_seed", base_seed + DEFAULT_TEST_SEED_OFFSET)) if test_seed is None else int(test_seed)

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

    input_scale = float(meta["input_scale"])
    test_x = torch.clamp(test_x / (2.0 * input_scale) + 0.5, 0.0, 1.0)

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

    np.savez(
        output_path,
        test_x=test_x.detach().cpu().numpy().astype(np.float32),
        test_y=test_y.detach().cpu().numpy().astype(np.int64),
        meta_json=json.dumps(meta_out, sort_keys=True),
    )
    return meta_out



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the exact MIMO test set from a training checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best.tar or checkpoint.tar from training")
    parser.add_argument("--output", required=True, help="Output .npz file path")
    parser.add_argument("--seed", type=int, default=None, help="Override base seed; defaults to checkpoint value or 1")
    parser.add_argument("--test_seed", type=int, default=None, help="Override test seed; defaults to checkpoint value or seed+1")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on exported samples")
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    meta = export_test_set(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        seed=args.seed,
        test_seed=args.test_seed,
        max_samples=args.max_samples,
    )

    print(f"Saved exact normalized MIMO test set to: {args.output}")
    print(json.dumps(meta, indent=2, sort_keys=True))
