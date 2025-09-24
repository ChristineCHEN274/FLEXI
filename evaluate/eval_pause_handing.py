import os
import json
import argparse
from typing import Dict, Optional, Tuple

def safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[Skip] Missing file: {path}")
    except json.JSONDecodeError as e:
        print(f"[Skip] Failed to parse JSON: {path} ({e})")
    except Exception as e:
        print(f"[Skip] Error reading file: {path} ({e})")
    return None


def compute_pause_window_with_model_latency(
    pause_json: dict, model_latency: float
) -> Optional[Tuple[float, float, float, float]]:
    
    ts = pause_json.get("timestamp")
    if not (isinstance(ts, list) and len(ts) >= 2 and ts[0] is not None and ts[1] is not None):
        return None

    pause_start = float(ts[0])
    pause_end_raw = float(ts[1])
    pause_duration = float(pause_end_raw - pause_start)
    if pause_duration < 0:
        return None

    effective_extra = max(model_latency - pause_duration, 0)

    pause_end_adj = pause_end_raw + effective_extra
    return pause_start, pause_end_adj, pause_duration, effective_extra

def first_chunk_start(output_json: dict) -> Optional[float]:
    chunks = output_json.get("chunks")
    if not (isinstance(chunks, list) and len(chunks) > 0):
        return None
    first = chunks[0]
    ts = first.get("timestamp")
    if not (isinstance(ts, list) and len(ts) >= 1 and ts[0] is not None):
        return None
    return float(ts[0])

def last_chunk_end(input_json: dict) -> Optional[float]:
    chunks = input_json.get("chunks")
    if not (isinstance(chunks, list) and len(chunks) > 0):
        return None
    last = chunks[-1]
    ts = last.get("timestamp")
    if not (isinstance(ts, list) and len(ts) >= 2 and ts[1] is not None):
        return None
    return float(ts[1])

def process_subfolder(subfolder_path: str) -> Optional[int]:

    pause_json_path = os.path.join(subfolder_path, "pause.json")
    input_json_path = os.path.join(subfolder_path, "input.json")
    output_json_path = os.path.join(subfolder_path, "output.json")

    pause_data = safe_load_json(pause_json_path)
    input_data = safe_load_json(input_json_path)
    output_data = safe_load_json(output_json_path)
    if pause_data is None or input_data is None or output_data is None:
        return None

    model_latency = fixed_latency

    pw = compute_pause_window_with_model_latency(pause_data, model_latency)
    if pw is None:
        print(f"[Skip] {subfolder_path}: invalid pause.json (missing timestamp or wrong order)")
        return None
    pause_start, pause_end_adj, pause_duration, effective_extra = pw

    first_ts = first_chunk_start(output_data)
    if first_ts is None:
        print(f"[Skip] {subfolder_path}: output.json missing first chunk timestamp")
        return None

    input_last_end = last_chunk_end(input_data)
    if input_last_end is None:
        print(f"[Skip] {subfolder_path}: input.json missing last chunk end timestamp")
        return None

    if first_ts > input_last_end:
        tor = 0
        return tor

    tor = 1 if (pause_start <= first_ts <= pause_end_adj) else 0
    return tor

def process_folder(root_folder: str) -> None:
    total = 0
    tor_sum = 0
    skipped = 0

    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            tor = process_subfolder(subfolder_path)
            if tor is None:
                skipped += 1
                continue
            total += 1
            tor_sum += tor

    print("=" * 60)
    print("[Result]")
    avg = (tor_sum / total) if total > 0 else 0.0
    print(f"\n  -TOR: {avg:.6f}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pause/input/output JSON folders")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of your dataset")
    parser.add_argument("--latency", type=float, default=0.0, help="Model latency")
    args = parser.parse_args()
    fixed_latency = args.latency
    process_folder(args.root_dir)