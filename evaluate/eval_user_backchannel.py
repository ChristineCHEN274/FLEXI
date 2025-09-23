import os
import json
import re
import argparse
from tqdm import tqdm
import numpy as np
import time
from openai import OpenAI

MODEL_NAME = "gpt-4-turbo"


def for_backchannel_time(file_path,file_path_metadata):
    getted = True
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        chunks = data.get('chunks', [])
        
        if len(chunks) < 2:
            print("Chunks list is empty or too short to calculate.")
            return None

        for i in range(1, len(chunks)):
            current_start = chunks[i]['timestamp'][0]
            previous_end = chunks[i - 1]['timestamp'][1]
            
            time_difference = current_start - previous_end
            if time_difference > 1.5:
                return float(chunks[i]['timestamp'][0])
        
        getted = False
        if not getted :
            with open(file_path_metadata, 'r', encoding='utf-8') as f:
                data = json.load(f)
            user_backchannel_times = data.get('user_backchannel_times', [])
            return user_backchannel_times[0]
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print("Invalid JSON file format")
    except Exception as e:
        print(f"Error occurred: {e}")

def extract_user_backchannel_times(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        user_backchannel_times = data.get('user_backchannel_times', [])
        
        if not user_backchannel_times:
            print(f"No 'user_backchannel_times' found in {json_file_path}")
            return []
        
        return float(user_backchannel_times[0])
    
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {json_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_last_end_ts(json_path: str, backchannel_time: float):
    """Get the last chunk end timestamp filtered by backchannel_time."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        try:
            chunks = data['chunks']
            for i in range(len(chunks) - 1, -1, -1):  
                if backchannel_time != 0:
                    last_chunk = chunks[i]
                    timestamp_last_element = last_chunk['timestamp'][1]
                    diff = abs(timestamp_last_element - backchannel_time)
                    if diff < 1.5 or timestamp_last_element >= backchannel_time:
                        continue  
                    else:
                        return float(timestamp_last_element)
                else:
                    last_chunk = data['chunks'][i]
                    timestamp_last_element = last_chunk['timestamp'][1]
                    return float(timestamp_last_element)
            print("No suitable chunk found")
            return None
        except Exception as e:
            print("Error occurred:", e)
            return None


def process_json(file_path,backchannel_time):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    chunks = data.get('chunks', [])
    if not chunks:
        print(f"No 'chunks' found in {file_path}")
        return None,None
    
    timestamps = [chunk['timestamp'] for chunk in chunks]
    differences = []
    
    for i in range(1, len(timestamps)):
        diff = timestamps[i][0] - timestamps[i-1][1]
        differences.append(diff)
    
    for i, diff in enumerate(differences):
        if diff > 1.5 and min(timestamps[i][0], timestamps[i-1][1]) <= backchannel_time:
            sentence1 = ' '.join(chunk['text'] for chunk in chunks[:i+1])
            sentence2 = ' '.join(chunk['text'] for chunk in chunks[i+1:])
            return sentence1, sentence2

    return None , None

def traverse_folder(folder_path):
    """Traverse dataset folder and collect user_backchannel splits."""
    result_back = {}
    cut_list = []
    for root, dirs, files in os.walk(folder_path):
        file_input = os.path.join(root, "input.json")
        file_metadata = os.path.join(root, "metadata.json")
        file_output = os.path.join(root, "output.json")

        if os.path.exists(file_input) and os.path.exists(file_output):
            backchannel_time = for_backchannel_time(file_input, file_metadata)
            if backchannel_time is None:
                continue

            last_input_time = get_last_end_ts(file_input, backchannel_time)
            result = process_json(file_output, backchannel_time)
            if result is not None:
                sentence1, sentence2 = result
                if sentence1 is not None:
                    result_back[root] = {"1": sentence1, "2": sentence2}
                    cut_list.append(root)
                else:
                    result_back[root] = 'empty'
            else:
                result_back[root] = 'empty'

    interruption_count = len(set(cut_list))
    interruption_rate = interruption_count / len(result_back) if result_back else 0
    interrupted_cases = set(cut_list)

    return result_back, interruption_count, interruption_rate, interrupted_cases

def parse_llm_rating(response_text: str) -> dict:
    try:
        json_str_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_str_match:
            print(f"Warning: No JSON block found in LLM response: {response_text}")
            return {}
        data = json.loads(json_str_match.group(0))
        return {
            "accuracy_score": float(data.get("accuracy_score", 0)),
            "analysis": data.get("analysis", "")
        }
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"Warning: Failed to parse LLM response: {e}\nOriginal: {response_text}")
        return {}


def eval_user_backchannel(source_folder):
    client = OpenAI(api_key= "YOUR_OPENAI_API_KEY")

    system_prompt = """
You are an impartial judge. Given "sentence1" and "sentence2", your task is to assess the semantic coherence between the two sentences and evaluate semantic_coherence.

[Output Requirement]
- Respond with a SINGLE JSON object that contains BOTH keys: "semantic_coherence" and "analysis".
- "semantic_coherence" ranges from 0–5 and may use 0.5 increments (e.g., 4.5).
- "analysis" should be 1–3 concise sentences explaining the judgment.

[Scoring Definition]
5 — Highly coherent: The two sentences are logically and contextually connected, forming a seamless and natural continuation. The relationship between the sentences is clear and strong.
4 — Moderately coherent: The two sentences are generally connected and form a logical sequence, but the connection may be somewhat weak or require a bit of inference.
3 — Slightly coherent: The two sentences have some connection, but the relationship is not very strong or may be somewhat forced.
2 — Minimally coherent: The two sentences have very little connection, and the relationship between them is unclear or weak.
1 — Incoherent: The two sentences have no logical or contextual connection and do not form a coherent sequence.
0 — Completely incoherent: The two sentences are entirely unrelated and do not form any meaningful connection.

Example STRICT format:
{
  "semantic_coherence": 5,
  "analysis": "The two sentences are highly coherent, forming a seamless and natural continuation."
}
"""

    accuracy_scores = []
    coherence_count = 0 

    # result_dict = traverse_folder(source_folder)
    result_dict, interruption_count, interruption_rate, interrupted_cases = traverse_folder(source_folder)
    total = 0
    count = 0
    for root, dirs, files in os.walk(source_folder):
        if root in result_dict:
            if result_dict[root] != 'empty':
                sentence1 = result_dict[root]['1']
                sentence2 = result_dict[root]['2']
                user_prompt = (
                    f"sentence1: \"{sentence1}\"\n\n"
                    f"sentence2: \"{sentence2}\""
                )
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0
                    )
                    prediction = response.choices[0].message.content
                except Exception as e:
                    print(f"API call failed for {root}: {e}")
                    continue

                parsed_output = parse_llm_rating(prediction)
                if parsed_output:
                    accuracy_scores.append(parsed_output['accuracy_score'])

                    with open(os.path.join(root, 'rating.json'), 'w', encoding='utf-8') as f:
                        json.dump(parsed_output, f, ensure_ascii=False, indent=2)
                    print(f"LLM rating result saved to rating.json for {root}")
                    accuracy_score = parsed_output.get('accuracy_score', None)
                    total+=accuracy_score
                    count+=1

                    if accuracy_score>2:
                        coherence_count += 1
                time.sleep(0.5) 

    print("---------------------------------------------------")
    print("[Result]")
    print("\n  -Interrupt Rate:", interruption_rate)
    if count != 0:
        print(f"\n  -Coherent cases: {coherence_count}/{count}")
    else:
        print("No LLM ratings computed.")
    print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate user_backchannel interruptions.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the dataset")
    args = parser.parse_args()

    eval_user_backchannel(args.root_dir)