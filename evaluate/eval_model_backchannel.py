import os
import json
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import interp1d

from silero_vad import load_silero_vad, get_speech_timestamps
import torchaudio
import argparse

window_size = 0.2 
epsilon = 1e-10
time_threshold = 3


def eval_backchannel(data_dir):

  
    gt_distribution_path = "/your/path/ai_gt_distribution.json"  # Obtained by get_gt_gpt.py
    with open(gt_distribution_path, "r") as f:
        gt_distribution = json.load(f)

    vad_model = load_silero_vad()
    jsd_list = []
    TOR_list = []
    freq_list = []
    for spk in os.listdir(data_dir):
        duration = 0

        out_wav_path = f"{data_dir}/{spk}/output.wav"
        if not os.path.exists(out_wav_path):
            raise FileNotFoundError("Required file 'output.wav' not found.")

        wav, sr = torchaudio.load(out_wav_path)

        target_sr = 16000
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr

        segments = get_speech_timestamps(
            wav,
            vad_model,
            return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )

        backchannel_prediction = []

        json_file_name = f"{data_dir}/{spk}/output.json"

        if not os.path.exists(json_file_name):
            print("The output.json file does not exist")
            backchannel_prediction = []

        else:
            TOR = 0
            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]
                duration = end_time - start_time
                print(f"duration is {duration}")

                if duration > time_threshold:  # means the model is formally speaking
                    TOR = 1
                    break
                else:
                    # short duration and possibly backchannel
                    # by the time-aligned trancription, get the text of the segment by using the start and end time
                    curr_text = []
                    with open(json_file_name, "r") as f:
                        data = json.load(f)
                        segments_cw = data["chunks"]

                        for segment_cw in segments_cw:
                            # Get timestamps, handling potential None/null values
                            t_start = segment_cw["timestamp"][0]
                            t_end = segment_cw["timestamp"][1]

                            # Skip segments with invalid timestamps
                            if t_start is None or (
                                t_end is None and t_start < end_time
                            ):
                                continue

                            # If end is None but start is valid, treat as potentially relevant
                            if t_end is None:
                                t_end = t_start  # Assume some duration for null-ended segments

                            # Check for any overlap with the target time range
                            if t_start >= start_time and t_end <= end_time:
                                curr_text.append(segment_cw["text"])
                            elif t_start <= end_time and t_end > end_time:
                                curr_text.append(segment_cw["text"])
                            elif t_start <= start_time and t_end > start_time:
                                curr_text.append(segment_cw["text"])

                    # check if the text contains backchannel
                    # 1. check the duration of the output speech,
                    num_words = len(curr_text)
                    if num_words > 3:
                        TOR = 1
                    else:
                        if duration < 1:
                            if len(curr_text) <= 2:
                                TOR = 0
                            else:
                                TOR = 1
                        else:
                            TOR = 1

                    if len(backchannel_prediction) > 0:
                        print(backchannel_prediction)

                    if TOR == 1:
                        backchannel_prediction = []
                    else:
                        backchannel_prediction.append([start_time, end_time])


        TOR_list.append(TOR)
        max_end_time = wav.shape[-1] / sr
        freq_list.append(len(backchannel_prediction) / max_end_time)

        if len(backchannel_prediction) == 0:
            js_divergence = 1
        else:
            time_intervals = [0 for i in range(int(max_end_time / window_size) + 1)]

            for interval in backchannel_prediction:
                start = int(interval[0] / window_size)
                end = int(interval[1] / window_size)
                for i in range(start, end + 1):
                    if i < len(time_intervals):
                        time_intervals[i] += 1

            time_intervals = np.array(time_intervals)
            time_intervals = time_intervals + epsilon  # Avoid division by zero
            time_intervals = time_intervals / np.sum(time_intervals)
            time_intervals = list(time_intervals)

            gt_dist = gt_distribution[spk]

            x_gt = np.linspace(0, 1, len(gt_dist))
            x_pred = np.linspace(0, 1, len(time_intervals))

            interp_func = interp1d(
                x_gt, gt_dist, kind="linear", fill_value="extrapolate"
            )
            gt_dist_resized = interp_func(x_pred)

            hist1 = np.array(time_intervals)
            hist2 = np.array(gt_dist_resized)

            js_divergence = jensenshannon(hist1, hist2)

        print(f"JSD: {js_divergence}")
        jsd_list.append(js_divergence)

    jsd_mean, jsd_std = np.mean(jsd_list), np.std(jsd_list)
    tor_mean, tor_std = np.mean(TOR_list), np.std(TOR_list)
    print("len(jsd_list)", len(jsd_list))
    print("len(TOR_list)", len(TOR_list))
    freq_mean, freq_std = np.mean(freq_list), np.std(freq_list)
    print("---------------------------------------------------")
    print("[Result]")
    print(f"\n  JSD : {jsd_mean:.4f}")
    print(f"\n  TOR : {tor_mean:.4f}")
    print(f"\n  BCR : {freq_mean:.4f}")
    print("---------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    eval_backchannel(args.root_dir)