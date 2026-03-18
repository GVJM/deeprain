"""
PrecipModels/make_queue.py — Queue generator for multi-dataset batch training.

Reads station count S from a .dat file, computes architecture sizes scaled to S,
and writes a TRAINING_QUEUE.json-compatible JSON file.

Usage (from PrecipModels/):
    # Generate queue for a new dataset, auto-detect S from file
    python make_queue.py --data_path ../dados_sabesp/data_precip.dat \\
                         --prefix dp24 \\
                         --output QUEUE_precip24.json

    # Append to existing queue (checks for duplicate variant_names)
    python make_queue.py --data_path ../dados_sabesp/data_precip.dat \\
                         --prefix dp24 \\
                         --append TRAINING_QUEUE.json

    # Include architecture sweep entries (Tier 1)
    python make_queue.py --data_path ../dados_sabesp/data_precip.dat \\
                         --prefix dp24 --tiers 0 1 \\
                         --output QUEUE_precip24.json

    # Override detected S
    python make_queue.py --data_path ../dados_sabesp/data_precip.dat \\
                         --prefix dp24 --n_stations 24 \\
                         --output QUEUE_precip24.json

Architecture sizing heuristics:
    hidden_size = clamp(nearest_pow2(S * 3), min=64,  max=512)
    gru_hidden  = hidden_size // 2
    rnn_hidden  = hidden_size // 2
    latent_size = clamp(nearest_pow2(S // 2), min=16, max=128)

| S   | hidden | gru/rnn | latent |
|-----|--------|---------|--------|
| 24  | 128    | 64      | 16     |
| 92  | 256    | 128     | 64     |  <- matches current SABESP defaults
"""

import argparse
import json
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Architecture sizing
# ──────────────────────────────────────────────────────────────────────────────

def _nearest_pow2(n: int) -> int:
    """Return the nearest power of 2 to n (rounds to closest)."""
    if n <= 0:
        return 1
    low = 1 << (n.bit_length() - 1)   # largest power of 2 <= n
    high = low << 1                     # smallest power of 2 >= n
    return high if (high - n) < (n - low) else low


def compute_arch(S: int) -> dict:
    """Return arch hyperparams for a dataset with S stations."""
    hidden_size = max(64, min(512, _nearest_pow2(S * 3)))
    latent_size = max(16, min(128, _nearest_pow2(S // 2)))
    rnn_hidden  = hidden_size // 2
    return dict(hidden_size=hidden_size, rnn_hidden=rnn_hidden, latent_size=latent_size)


# ──────────────────────────────────────────────────────────────────────────────
# Station count detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_stations(data_path: str) -> int:
    """Read the header line of a .dat file and count station columns."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with open(path) as f:
        header = f.readline().strip()
    # Support comma or tab/space separation
    if "," in header:
        cols = header.split(",")
    else:
        cols = header.split()
    # First column is the date/datetime column; remaining are stations
    n_stations = len(cols) - 1
    if n_stations <= 0:
        raise ValueError(f"Could not detect station columns in header: {header[:120]}")
    return n_stations


# ──────────────────────────────────────────────────────────────────────────────
# Entry builders per tier
# ──────────────────────────────────────────────────────────────────────────────

# Models that use gru_hidden (GRU-based context encoder)
_GRU_MODELS = {"ar_vae", "ar_flow_match", "ar_latent_fm", "ar_real_nvp", "ar_glow"}

# Models that use rnn_hidden (linear/MLP context)
_RNN_MODELS = {"ar_mean_flow", "ar_flow_map"}

# Models with a meaningful latent space (latent_size > 0 in MODEL_DEFAULTS)
_LATENT_MODELS = {"ar_vae", "ar_latent_fm"}

# Tier 0: one entry per base AR model
_TIER0_MODELS = [
    "ar_vae",
    "ar_flow_match",
    "ar_latent_fm",
    "ar_real_nvp",
    "ar_glow",
    "ar_mean_flow",
    "ar_flow_map",
]


def _base_entry(model: str, prefix: str, data_path: str, arch: dict) -> dict:
    """Build one Tier-0 queue entry for `model`."""
    entry = {
        "variant_name": f"{prefix}_{model}",
        "model":        model,
        "data_path":    data_path,
    }
    if model in _GRU_MODELS:
        entry["gru_hidden"] = arch["rnn_hidden"]   # gru_hidden = hidden_size // 2
    elif model in _RNN_MODELS:
        entry["rnn_hidden"] = arch["rnn_hidden"]
    entry["hidden_size"] = arch["hidden_size"]
    if model in _LATENT_MODELS:
        entry["latent_size"] = arch["latent_size"]
    return entry


def build_tier0(prefix: str, data_path: str, arch: dict) -> list:
    return [_base_entry(m, prefix, data_path, arch) for m in _TIER0_MODELS]


def build_tier1(prefix: str, data_path: str, arch: dict) -> list:
    """Architecture sweep entries scaled to dataset size."""
    H = arch["hidden_size"]
    # Three hidden sizes: H//2, H, H*2 — clamped to [64, 512]
    hidden_sizes = sorted({max(64, min(512, H // 2)), H, min(512, H * 2)})
    n_layers_list = [2, 4, 6]
    entries = []

    sweep_models = ["ar_mean_flow", "ar_flow_map", "ar_flow_match"]
    for model in sweep_models:
        for h in hidden_sizes:
            for nl in n_layers_list:
                vname = f"{prefix}_{model}_h{h}_l{nl}"
                entry = {
                    "variant_name": vname,
                    "model":        model,
                    "data_path":    data_path,
                    "hidden_size":  h,
                    "n_layers":     nl,
                }
                if model in _GRU_MODELS:
                    entry["gru_hidden"] = h // 2
                elif model in _RNN_MODELS:
                    entry["rnn_hidden"] = h // 2
                entries.append(entry)

    # VAE size sweep (deduplicated)
    vae_combos = {
        (max(64, min(512, H // 2)), max(16, min(128, arch["latent_size"] // 2))),
        (H, arch["latent_size"]),
        (min(512, H * 2), min(128, arch["latent_size"] * 2)),
    }
    for h, lat in sorted(vae_combos):
        vname = f"{prefix}_ar_vae_h{h}_lat{lat}"
        entries.append({
            "variant_name": vname,
            "model":        "ar_vae",
            "data_path":    data_path,
            "gru_hidden":   h // 2,
            "hidden_size":  h,
            "latent_size":  lat,
        })

    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate a TRAINING_QUEUE.json-compatible queue file for a given dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data_path",  required=True,
                   help="Path to .dat dataset file (used in queue entries and for station detection)")
    p.add_argument("--prefix",     required=True,
                   help="Short prefix for variant names (e.g. 'dp24')")
    p.add_argument("--n_stations", type=int, default=None,
                   help="Override station count instead of auto-detecting from file header")
    p.add_argument("--tiers",      nargs="+", type=int, default=[0],
                   choices=[0, 1],
                   help="Which tier(s) to generate. 0=base models, 1=arch sweep (default: 0)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", metavar="FILE",
                       help="Write new queue JSON to this file")
    group.add_argument("--append", metavar="FILE",
                       help="Append entries to existing queue file (skips duplicate variant_names)")
    args = p.parse_args()

    # Detect or use override S
    if args.n_stations is not None:
        S = args.n_stations
        print(f"Using override: S={S} stations")
    else:
        S = detect_stations(args.data_path)
        print(f"Detected S={S} stations from {args.data_path}")

    arch = compute_arch(S)
    print(f"Architecture: hidden_size={arch['hidden_size']}, "
          f"gru/rnn_hidden={arch['rnn_hidden']}, latent_size={arch['latent_size']}")

    # Build entries
    new_entries = []
    if 0 in args.tiers:
        t0 = build_tier0(args.prefix, args.data_path, arch)
        new_entries.extend(t0)
        print(f"Tier 0: {len(t0)} entries")
    if 1 in args.tiers:
        t1 = build_tier1(args.prefix, args.data_path, arch)
        new_entries.extend(t1)
        print(f"Tier 1: {len(t1)} entries")

    print(f"Total new entries: {len(new_entries)}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(new_entries, indent=2))
        print(f"Written to {out_path}")

    else:  # --append
        app_path = Path(args.append)
        if app_path.exists():
            with open(app_path) as f:
                existing = json.load(f)
        else:
            existing = []

        existing_names = {e["variant_name"] for e in existing}
        added = []
        skipped = []
        for entry in new_entries:
            vn = entry["variant_name"]
            if vn in existing_names:
                skipped.append(vn)
            else:
                existing.append(entry)
                added.append(vn)

        if skipped:
            print(f"Skipped {len(skipped)} duplicate variant(s): {skipped}")
        app_path.write_text(json.dumps(existing, indent=2))
        print(f"Added {len(added)} entries to {app_path}")


if __name__ == "__main__":
    main()
