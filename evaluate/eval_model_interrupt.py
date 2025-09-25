import os
import json
import re
import argparse
from tqdm import tqdm
import numpy as np
import time
from openai import OpenAI


MODEL_NAME = "gpt-4-turbo"      

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
        print(f"Warning: Failed to parse LLM response: {e}\nOriginal response: {response_text}")
        return {}


def get_last_end_ts(json_path: str):
    """Get the end timestamp of the last chunk from input.json."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    chunks = d.get("chunks", [])
    if not chunks or "timestamp" not in chunks[-1] or not chunks[-1]["timestamp"]:
        return None
    return float(chunks[-1]["timestamp"][-1])


def get_first_start_ts(json_path: str):
    """Get the start timestamp of the first chunk from output.json."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    chunks = d.get("chunks", [])
    if not chunks or "timestamp" not in chunks[0] or not chunks[0]["timestamp"]:
        return None
    return float(chunks[0]["timestamp"][0])


def eval_model_interruption(root_dir: str):
    client = OpenAI(api_key= "YOUR_API_KEY")

    system_prompt = """
You are an impartial judge. Given the "user's incorrect statement" and the "AI's response", your task is to assess whether the AI’s correction of the error is factually correct and sufficient, and evaluate accuracy_score.

[Output Requirement]
- Respond with a SINGLE JSON object that contains BOTH keys: "accuracy_score" and "analysis".
- "accuracy_score" ranges from 0–5 and may use 0.5 increments (e.g., 4.5).
- "analysis" should be 1–3 concise sentences explaining the judgment.

[Scoring Definition]
5 — Completely correct and comprehensive: all key facts are accurate; explicitly identifies the original error; provides necessary and appropriate justification/data/definitions; introduces no new errors.
4 — Essentially correct: the main conclusion is correct and sufficient to resolve the original error; may miss minor points or provide minimal justification; no major factual mistakes.
3 — Partially correct: the correction is on the right track but incomplete or with minor inaccuracies/omissions.
2 — Multiple inaccuracies: core correction shows clear mistakes/confusion, or lacks important facts making it potentially misleading.
1 — Mostly wrong: most key information is incorrect or the understanding of the error is clearly off.
0 — Completely wrong or no factual correction.

Example STRICT format:
{
  "accuracy_score": 5,
  "analysis": "The AI correctly identified the user's error and provided the right facts."
}
"""

    accuracy_scores = []
    durations = []
    interruption_count = 0  

    sample_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for sample_id in tqdm(sorted(sample_dirs), desc="LLM Judging Model Interruption"):
        sample_path = os.path.join(root_dir, sample_id)

        input_json_path = os.path.join(sample_path, "input.json")
        output_json_path = os.path.join(sample_path, "output.json")
        source_json_path = os.path.join(sample_path, "source_statement.json")
        rating_json_path = os.path.join(sample_path, "rating.json")


        input_end = get_last_end_ts(input_json_path)
        output_start = get_first_start_ts(output_json_path)
        if input_end is not None and output_start is not None:
            duration = output_start - input_end
            durations.append(duration)
            if duration < -0.5:
                interruption_count += 1
 
        if os.path.exists(rating_json_path):
            try:
                with open(rating_json_path, 'r', encoding='utf-8') as f:
                    rating_data = json.load(f)
                    if 'accuracy_score' in rating_data:
                        accuracy_scores.append(rating_data['accuracy_score'])
                    else:
                        print(f"Warning: {rating_json_path} missing 'accuracy_score' field.")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse {rating_json_path}, file may be corrupted or invalid. Error: {e}")
            continue 

        
        if not os.path.exists(output_json_path) or not os.path.exists(source_json_path):
            continue
        
        with open(output_json_path, 'r', encoding='utf-8') as f:
            ai_response_text = json.load(f).get("text", "")

        
        with open(source_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                user_statement_text = data[0].get("user_statement", "")
            else:
                user_statement_text = data.get("user_statement", "")

        if not ai_response_text or not user_statement_text:
            continue

        user_prompt = (
            f"User's incorrect statement: \"{user_statement_text}\"\n\n"
            f"AI's response: \"{ai_response_text}\""
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
            print(f"API call failed for {sample_id}: {e}")
            continue

        parsed_output = parse_llm_rating(prediction)
        if parsed_output and 'accuracy_score' in parsed_output:
            accuracy_scores.append(parsed_output['accuracy_score'])
            
            with open(rating_json_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_output, f, ensure_ascii=False, indent=2)

        time.sleep(0.5)  

    print("\n---------------------------------------------------")
    print("[Results]")
    if accuracy_scores:
        print(f"\n  -EDS: {np.mean(accuracy_scores):.2f}/5.0")
    else:
        print("\n  -EDS: N/A")
    if durations:
        avg_delay = float(np.mean(durations))
        print(f"\n  -TOR: {interruption_count}")
    else:
        print("\n  -TOR: 0")
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Proactive Interruption Evaluation using OpenAI Judge")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the model_interruption dataset.")
    args = parser.parse_args()
    eval_model_interruption(args.root_dir)
