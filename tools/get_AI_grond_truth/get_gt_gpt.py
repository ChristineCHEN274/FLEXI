import os
import json
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
import time
import re
from openai import OpenAI

WINDOW_SIZE = 0.2
EPSILON = 1e-10

def get_timestamps_from_llm(client: OpenAI, model_name: str, word_chunks: list, max_retries: int = 3) -> list[float]:

    transcript_with_time = "\n".join([f"'{chunk['text']}' (ends at {chunk['timestamp'][1]:.2f}s)" for chunk in word_chunks])
    
    system_prompt = """
    You are an expert dialogue analyst. Your task is to analyze a spoken monologue and identify the most natural moments for a listener to provide a short backchannel, such as "uh-huh", "yeah", or "I see". These moments typically occur at the end of a thought, a clause, or a sentence, often marked by a natural pause.

    You will be given a transcript where each word has an end timestamp. Your response MUST be a JSON object containing a single key "timestamps", with a value being a list of floats representing the exact end times of the words where a backchannel is appropriate.

    - The timestamps should correspond to the end of a word, right before a natural pause.
    - Only select moments where a brief acknowledgment would feel natural and not interruptive.
    - If no moments are suitable, return an empty list: {"timestamps": []}.
    - Your entire response should be ONLY the JSON object, with no other text before or after it.

    Example:
    User Input:
    'I' (ends at 0.52s)
    'went' (ends at 0.78s)
    'park' (ends at 1.45s)
    ',' (ends at 2.15s)

    Your Correct Response:
    {"timestamps": [2.15]}
    """

    user_prompt = f"Here is the transcript, please analyze it and provide the backchannel timestamps in the required JSON format:\n\n{transcript_with_time}"
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                json_string = json_match.group(0)
                try:
                    json_response = json.loads(json_string)
                    timestamps = json_response.get("timestamps", [])
                    if isinstance(timestamps, list) and all(isinstance(t, (int, float)) for t in timestamps):
                        return [float(t) for t in timestamps]
                    else:
                        print(f"Warning: LLM returned improperly formatted timestamps: {timestamps}")
                        continue
                except json.JSONDecodeError:
                    print(f"Warning: Extracted JSON block but failed to parse: '{json_string}'")
                    continue
            else:
                print(f"Warning: No valid JSON block found in LLM response: '{content}'")
                continue

        except Exception as e:
            print(f"Error: Exception occurred while calling or parsing LLM response (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)

    return []

def generate_llm_based_distribution(data_dir: str, output_path: str, model_name: str, api_key: str):

    client = OpenAI(api_key=api_key)

    gt_distributions = {}
    asr_files = sorted([os.path.join(root, "input.json") for root, _, files in os.walk(data_dir) if "input.json" in files])
    if not asr_files:
        print(f"Error: No 'input.json' files found under '{data_dir}'. Please run asr.py on input.wav first.")
        return
    print(f"Found {len(asr_files)} transcription files. Starting LLM-based auto-annotation...")
    for asr_path in tqdm(asr_files, desc="LLM Auto-Annotation"):

        spk_id = os.path.basename(os.path.dirname(asr_path))
        audio_path = asr_path.replace("input.json", "input.wav")

        if not os.path.exists(audio_path): continue

        with open(asr_path, 'r') as f:
            data = json.load(f)

        llm_timestamps = get_timestamps_from_llm(client, model_name, data.get("chunks", []))
        
        # Map LLM timestamps onto a fixed time grid of 0.2s and create a distribution vector
        metadata = torchaudio.info(audio_path)

        # Compute audio duration and number of bin
        total_duration = metadata.num_frames / metadata.sample_rate
        num_bins = int(total_duration / WINDOW_SIZE) + 1
        time_bins = np.zeros(num_bins)

        for timestamp in llm_timestamps:
            bin_index = int(timestamp / WINDOW_SIZE)
            if bin_index < num_bins:
                time_bins[bin_index] += 1

        # If LLM returned no timestamps, use a uniform distribution
        if np.sum(time_bins) == 0:
            time_bins = np.ones(num_bins)

        # Smooth and normalize
        normalized_bins = (time_bins + EPSILON) / np.sum(time_bins + EPSILON)

        gt_distributions[spk_id] = normalized_bins.tolist()

        time.sleep(1)

    with open(output_path, 'w') as f:
        json.dump(gt_distributions, f, indent=4)
    print(f"\nSuccess! LLM-based ground truth distribution file has been saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatically generate ground truth distribution JSON for backchannel detection using an LLM.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of the backchannel dataset (must contain input.json files).")
    parser.add_argument("--api_key", type=str,default="YOUR_API_KEY" , help="OpenAI API Key.")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo", help="OpenAI model name to use, e.g., gpt-4o, gpt-3.5-turbo.")
    parser.add_argument("--output_path", type=str, default="./ai_gt_distribution.json", help="Path to save the output JSON file.")
    args = parser.parse_args()
    generate_llm_based_distribution(args.data_dir, args.output_path, args.model_name, args.api_key)