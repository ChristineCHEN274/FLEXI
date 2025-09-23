import json
import re
import os
import argparse
from tqdm import tqdm
from openai import OpenAI

EPS = 0.15
TURN_DURATION_THRESHOLD = 1.0
TURN_NUM_WORDS_THRESHOLD = 3

def parse_output(data):
    example_pattern = re.compile(
        r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)", re.DOTALL
    )
    match = example_pattern.search(data)
    if match:
        analysis = match.group(1).strip()
        rating = match.group(2).strip()
        return {"analysis": analysis, "rating": int(rating)}
    return {}

def join_text_from_segments(segments):
    return " ".join(
        (c.get("text") or "").strip() for c in segments if (c.get("text") or "").strip()
    ).strip()

client = OpenAI(api_key="YOUR_API_KEY")

def eval_user_interruption(root_dir, client):
    MODEL_NAME = "gpt-4-turbo"
    seed = 0

    system_msg = """
    The scenario is that the user and AI are talking in the spoken conversation.
    The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
    Your task is to rate the quality of AI's response after the user interrupt the turn.

    Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
    - 0: The AI's response is totally unrelated to the user's interrupting turn.
    - 1: The AI's response is not related to the user's interrupting turn.
    - 2: The AI's response is slightly related to the user's interrupting turn.
    - 3: The AI's response is related to the user's interrupting turn.
    - 4: The AI's response is highly related to the user's interrupting turn.
    - 5: The AI's response is perfectly related to the user's interrupting turn.

    Firstly, briefly analyze the user's interrupting turn and the AI's response
    Then, you must return the overall output as the following format:
    Analysis: [Your analysis].
    I would rate the AI's response as [Rating].
    """

    file_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if "interrupt.json" in files and "output.json" in files:
            file_dirs.append(root)

    total_samples = 0
    tor_list = []
    latency_list = []
    llm_score_list = []
    
    tor_0_count = 0
    tor_0_and_responded_count = 0
    count_Intertwined = 0
    count_total_Intertwined=0
    for file_dir in tqdm(sorted(file_dirs)):
        print(f"Processing {file_dir} ...")
        total_samples += 1

        try:
            with open(os.path.join(file_dir, "output.json"), "r") as f:
                out_after_interrupt = json.load(f)
            with open(os.path.join(file_dir, "interrupt.json"), "r") as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  [Error] Skipping due to file error: {e}")
            continue

        ui_start, ui_end = metadata[0]["timestamp"]
        all_ai_segments = out_after_interrupt.get("chunks", [])

        is_speaking_during_interrupt = True
        Intertwined_start = 0
        Intertwined_end = 0
        for segment in all_ai_segments:
            seg_start, seg_end = segment["timestamp"]
            if seg_start < ui_end and seg_end > ui_start:
                if Intertwined_start == 0:
                    Intertwined_start = seg_start
                    Intertwined_end = seg_end
                else:
                    Intertwined_end = seg_end
                continue
            if seg_start > ui_end or seg_end > ui_end:
                if seg_start - Intertwined_end > 1:
                    count_Intertwined+=(Intertwined_end - Intertwined_start)
                    count_total_Intertwined+=1
                    if Intertwined_end - Intertwined_start < 3:
                        is_speaking_during_interrupt = False
                        break
                    else:
                        is_speaking_during_interrupt = True
                        break

        TOR = 1 if is_speaking_during_interrupt else 0
        tor_list.append(TOR)

        segments_after_interrupt = [
            c for c in all_ai_segments if c["timestamp"][0] >= ui_end + EPS
        ]
        
        has_new_response = False
        new_response_text = ""
        if segments_after_interrupt:
            new_response_text = join_text_from_segments(segments_after_interrupt)
            duration = segments_after_interrupt[-1]["timestamp"][-1] - segments_after_interrupt[0]["timestamp"][0]
            if duration >= TURN_DURATION_THRESHOLD or len(new_response_text.split()) > TURN_NUM_WORDS_THRESHOLD:
                has_new_response = True

        llm_score = 0
        
        if TOR == 1:
            print(f"  [Result] TOR=1. Model did not yield. Final Score = 0.")
            llm_score = 0
        else:
            tor_0_count += 1
            if has_new_response:
                tor_0_and_responded_count += 1
                
                output_start_time = segments_after_interrupt[0]["timestamp"][0]
                latency = output_start_time - ui_end
                latency_list.append(latency)

                print(f"  [Result] TOR=0, has new response. Latency={latency:.2f}s. Evaluating with LLM...")
                
                rating_path = os.path.join(file_dir, "rating.json")
                if os.path.exists(rating_path):
                    print(f"    Found existing rating.json, skipping API call.")
                    try:
                        with open(rating_path, "r") as f:
                            rating_data = json.load(f)
                        llm_score = rating_data.get('rating', 0)
                    except json.JSONDecodeError:
                        print(f"    Warning: Could not parse {rating_path}, scoring as 0.")
                        llm_score = 0
                else:
                    in_interrupt_text = metadata[0]["interrupt"]
                    in_before_interrupt_text = metadata[0]["context"]
                    
                    user_msg = f"""
                    - Contextual user turn: {in_before_interrupt_text}
                    - User interrupting turn: {in_interrupt_text}
                    - AI's response: {new_response_text}
                    """
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ]
                    
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            seed=seed,
                        )
                        prediction = response.choices[0].message.content
                        parsed_output = parse_output(prediction)
                        print(f"    LLM Analysis: {parsed_output.get('analysis')}")
                        print(f"    LLM Rating: {parsed_output.get('rating')}")
                        
                        if "rating" in parsed_output:
                            llm_score = parsed_output["rating"]
                            with open(rating_path, "w") as f:
                                json.dump(parsed_output, f, indent=4)
                        else:
                            print("    Warning: LLM did not return a valid rating. Scoring as 0.")
                            llm_score = 0
                    except Exception as e:
                        print(f"    [Error] OpenAI API call failed: {e}. Scoring as 0.")
                        llm_score = 0
            else:
                print(f"  [Result] TOR=0, but no new response. Final Score = 0.")
                print("TOR=0 with no response path:",file_dir)
                llm_score = 0
        
        llm_score_list.append(llm_score)

    yield_rate = 1 - (sum(tor_list) / len(tor_list)) if tor_list else 0
    avg_llm_rating = sum(llm_score_list) / len(llm_score_list) if llm_score_list else 0
    avg_llm_rating_tor_0 =  (avg_llm_rating*total_samples)/tor_0_count
    print("\n---------------------------------------------------")
    print("[Results]")
    print(f"\n  -TTR: {yield_rate:.2%}")
    print(f"\n  -LLM Rating: ",avg_llm_rating_tor_0)
    print(f"\n  -Latency: ",count_Intertwined/count_total_Intertwined,"s")
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate user interruption scenarios with new metrics.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing evaluation case subdirectories.")
    args = parser.parse_args()
    eval_user_interruption(args.root_dir, client)
