import os
import json
import argparse
from glob import glob

import soundfile as sf
import nemo.collections.asr as nemo_asr
from tqdm import tqdm

MODEL_NAME = ""


def get_time_aligned_transcription(data_path):
    # Collect all output.wav files under the root directory
    audio_paths = sorted(glob(f"{data_path}/*/{MODEL_NAME}output.wav"))

    # Load the pretrained NeMo ASR model and move to GPU
    local_model_path = "./model/parakeet-tdt-0.6b-v2.nemo"
    asr_model = nemo_asr.models.ASRModel.restore_from(local_model_path).cuda()

    for audio_path in tqdm(audio_paths):
        print(audio_path)
        # Read the audio file (waveform and sample rate)
        waveform, sr = sf.read(audio_path)
        # If multichannel audio, convert to mono by averaging channels
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sr)
            # original file‐based API (this accepts timestamps=True)
            asr_outputs = asr_model.transcribe([tmp.name], timestamps=True)
        # remove the temp file so you don't leak disk
        os.unlink(tmp.name)

        # Take the first (and only) result
        result = asr_outputs[0]
        word_timestamps = result.timestamp["word"]

        # Build the output dict
        chunks = []
        text = ""
        for w in word_timestamps:
            start_time = w["start"]
            end_time = w["end"]
            word = w["word"]

            text += word + " "
            chunks.append(
                {
                    "text": word,
                    "timestamp": [start_time, end_time],
                }
            )

        output_dict = {
            "text": text.strip(),
            "chunks": chunks,
        }

        # Write the JSON result next to the WAV file
        result_path = audio_path.replace(f"{MODEL_NAME}output.wav", "output.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe full audio"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root folder containing subfolders with output.wav",
    )
    args = parser.parse_args()

    get_time_aligned_transcription(args.root_dir)
