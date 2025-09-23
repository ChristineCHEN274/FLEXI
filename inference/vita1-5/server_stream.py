from __future__ import print_function
import warnings

import atexit
import argparse
import asyncio
import base64
import builtins
import datetime
import io
import json
import multiprocessing
import os
import re
import cv2
import threading
import time
import torchaudio
import torch
from queue import Empty
from typing import AsyncGenerator
from threading import Timer

import wave

import numpy as np
from PIL import Image
from flask import Flask, current_app, render_template, request,jsonify
from flask_socketio import SocketIO, disconnect, emit

from web_demo.vita_html.web.parms import GlobalParams
from web_demo.vita_html.web.pem import generate_self_signed_cert

TTS_SR = 24000  
REC_STATE = {}  # { sid: {"running":bool, "t0":float, "fh":wave.Wave_write, "written":int} }

warnings.filterwarnings("ignore", message=r".*torch\.library\.impl_abstract.*")
warnings.filterwarnings("ignore", message=r".*timm\.models\.layers.*")
warnings.filterwarnings("ignore", message=r".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=r".*mamba_ssm.*")
warnings.filterwarnings("ignore", message=r".*Nvidia apex.*")

def get_args():
    parser = argparse.ArgumentParser(description='VITA')
    parser.add_argument('--model_path', help='model_path to load', default='/your/path/VITA-main/demo_VITA_ckpt')
    parser.add_argument('--ip', help='ip of server', default='127.0.0.1')
    parser.add_argument('--port', help='port of server', default=8082)
    parser.add_argument('--max_users', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=6000)
    args = parser.parse_args()
    print(args)
    return args

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    original_print(f'[{current_time}]', *args, **kwargs)

args = get_args()

decoder_topk = 2
codec_padding_size = 10
target_sample_rate = 16000
last_tts_model_id = 0

IMAGE_TOKEN_INDEX = 51000
AUDIO_TOKEN_INDEX = 51001
IMAGE_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"
VIDEO_TOKEN = "<video>"

# change print function to add time stamp
original_print = builtins.print
builtins.print = custom_print

# init flask app
app = Flask(__name__, template_folder='./vita_html/web/resources', static_folder='./vita_html/web/static')
socketio = SocketIO(app)
# init connected users
connected_users = {}


def disconnect_user(sid):
    if sid in connected_users:  
        stop_tts_recording(sid)

        print(f"Disconnecting user {sid} due to time out")
        socketio.emit('out_time', to=sid) 
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]

def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
        except Empty:
            break

def mark_interrupt(sid, *, bump_run=False, clear_record_queue=False):
    now = time.monotonic()
    st = REC_STATE.get(sid)
    if st is not None:
        st["cut_ts"] = now
    if bump_run:
        app.config['RUN_ID'].value += 1
        if st is not None:
            st["run_id"] = app.config['RUN_ID'].value
    if clear_record_queue:
        try:
            clear_queue(current_app.config['TTS_RECORD_QUEUE'])
        except Exception:
            pass

def load_model_embemding(model_path):
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    config_path = os.path.join(model_path, 'origin_config.json')
    config = VITAQwen2Config.from_pretrained(config_path)
    model = VITAQwen2ForCausalLM.from_pretrained(model_path, config=config, low_cpu_mem_usage=True)
    embedding = model.get_input_embeddings()
    del model
    return embedding

def save_video(images, video_filename):
    if len(images) == 0:
        return
        
    copy_images = list(images)
    height, width, layers = copy_images[0].shape
    size = (width, height)
    print(f"Saving video with size {size}")

    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
    for image in copy_images:
        out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    out.release()

# This is a function to tokenize the prompt with image and audio tokens
def tokenizer_image_audio_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = []
    for chunk in re.split(r'(<audio>|<image>)', prompt):
        if chunk == '<audio>':
            prompt_chunks.append([audio_token_index])
        elif chunk == '<image>':
            prompt_chunks.append([image_token_index])
        else:
            prompt_chunks.append(tokenizer(chunk).input_ids)
    
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in prompt_chunks:
        if x != [image_token_index] and x != [audio_token_index]:
            input_ids.extend(x[offset:])
        else:
            input_ids.extend(x[:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.LongTensor(input_ids)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def load_model(
        llm_id,
        engine_args,
        cuda_devices,
        inputs_queue,
        outputs_queue,
        tts_outputs_queue,
        stop_event,
        other_stop_event,
        worker_ready,
        wait_workers_ready,
        start_event,
        other_start_event,
        start_event_lock,
        global_history,
        global_history_limit=0,
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    import torch
    import torchaudio
    from vllm import LLM, SamplingParams
    from transformers import AutoFeatureExtractor, AutoTokenizer
    from decord import VideoReader, cpu
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    
    print(wait_workers_ready,'wait_workers_readywait_workers_readywait_workers_ready')
    wait_workers_ready[1].wait()
    print(wait_workers_ready,'wait_workers_readywait_workers_ready')
    llm = LLM(
            model=engine_args,
            dtype="float16",
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,  
            disable_custom_all_reduce=True,
            limit_mm_per_prompt={'image':256,'audio':50}
        )

    tokenizer = AutoTokenizer.from_pretrained(engine_args, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(engine_args, subfolder="feature_extractor", trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0.001, max_tokens=512, best_of=1, skip_special_tokens=False)

    def _process_inputs(inputs):

        def _process_image(image_path):
            if isinstance(image_path, str):
                assert os.path.exists(image_path), f"Image file {image_path} does not exist."
                return Image.open(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
            else:
                assert isinstance(image_path, np.ndarray), "Image must be either a file path or a numpy array."
                return Image.fromarray(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)


        def _process_audio(audio_path):
            assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist."
            audio, sr = torchaudio.load(audio_path)
            audio_features = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
            audio_features = audio_features.squeeze(0)
            return audio_features
        
        def _process_video(video_path, max_frames=4, min_frames=4, s=None, e=None):
            # speed up video decode via decord.

            if s is None or e is None:
                start_time, end_time = None, None
            else:
                start_time = int(s)
                end_time = int(e)
                start_time = max(start_time, 0)
                end_time = max(end_time, 0)
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = start_time + 1

            if os.path.exists(video_path):
                vreader = VideoReader(video_path, ctx=cpu(0))
            else:
                raise FileNotFoundError(f"Video file {video_path} does not exist.")

            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            
            if num_frames > 0:
                # T x 3 x H x W
                all_pos = list(range(f_start, f_end + 1))
                if len(all_pos) > max_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
                elif len(all_pos) < min_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
                else:
                    sample_pos = all_pos

                # patch_images = [Image.fromarray(f).transpose(Image.FLIP_LEFT_RIGHT) for f in vreader.get_batch(sample_pos).asnumpy()]
                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
                return patch_images

            else:
                print("video path: {} error.".format(video_path))

        if "multi_modal_data" in inputs:

            if "image" in inputs["multi_modal_data"]:
                image_inputs = inputs["multi_modal_data"]["image"]
                if not isinstance(image_inputs, list):
                    image_inputs = [image_inputs]
                inputs["multi_modal_data"]["image"] = [_process_image(f) for f in image_inputs]

                if "prompt" in inputs:
                    assert inputs["prompt"].count(IMAGE_TOKEN) == len(image_inputs), \
                        f"Number of image token {IMAGE_TOKEN} in prompt must match the number of image inputs."
                elif "prompt_token_ids" in inputs:
                    assert inputs["prompt_token_ids"].count(IMAGE_TOKEN_INDEX) == len(image_inputs), \
                        f"Number of image token ids {IMAGE_TOKEN_INDEX} in prompt_token_ids must match the number of image inputs."
                else:
                    raise ValueError("Either 'prompt' or 'prompt_token_ids' must be provided.")

            if "audio" in inputs["multi_modal_data"]:
                audio_inputs = inputs["multi_modal_data"]["audio"]
                if not isinstance(audio_inputs, list):
                    audio_inputs = [audio_inputs]
                inputs["multi_modal_data"]["audio"] = [_process_audio(f) for f in audio_inputs]

                if "prompt" in inputs:
                    assert inputs["prompt"].count(AUDIO_TOKEN) == len(inputs["multi_modal_data"]["audio"]), \
                        f"Number of audio token {AUDIO_TOKEN} in prompt must match the number of audio inputs."
                elif "prompt_token_ids" in inputs:
                    assert inputs["prompt_token_ids"].count(AUDIO_TOKEN_INDEX) == len(inputs["multi_modal_data"]["audio"]), \
                        f"Number of audio token ids {AUDIO_TOKEN_INDEX} in prompt_token_ids must match the number of audio inputs."
                else:
                    raise ValueError("Either 'prompt' or 'prompt_token_ids' must be provided.")

            if "video" in inputs["multi_modal_data"]:
                video_inputs = inputs["multi_modal_data"]["video"]
                if not isinstance(video_inputs, list):
                    video_inputs = [video_inputs]

                assert "prompt" in inputs, "Prompt must be provided when video inputs are provided."
                assert "image" not in inputs["multi_modal_data"], "Image inputs are not supported when video inputs are provided."

                assert inputs["prompt"].count(VIDEO_TOKEN) == 1, "Currently only one video token is supported in prompt."

                assert inputs["prompt"].count(VIDEO_TOKEN) == len(inputs["multi_modal_data"]["video"]), \
                    f"Number of video token {VIDEO_TOKEN} in prompt must match the number of video inputs."
                
                video_frames_inputs = []
                for video_input in video_inputs:
                    video_frames_inputs.extend(_process_video(video_input, max_frames=4, min_frames=4))
                inputs["prompt"] = inputs["prompt"].replace(VIDEO_TOKEN, IMAGE_TOKEN * len(video_frames_inputs))
                if "image" not in inputs["multi_modal_data"]:
                    inputs["multi_modal_data"]["image"] = []
                inputs["multi_modal_data"]["image"].extend(video_frames_inputs)

                inputs["multi_modal_data"].pop("video", None)

        return inputs

    def judge_negative(text):
        is_negative = text.startswith('☟')
        return is_negative
    

    async def stream_results(results_generator) -> AsyncGenerator[bytes, None]:
        previous_text = ""
        async for request_output in results_generator:

            text = request_output.outputs[0].text
            newly_generated_text = text[len(previous_text):]
            previous_text = text
            yield newly_generated_text

    async def collect_results_demo(results_generator):
        async for newly_generated_text in stream_results(results_generator):
            continue

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    worker_ready.set()
    if not isinstance(wait_workers_ready, list):
        wait_workers_ready = [wait_workers_ready]

    while True:
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        if not inputs_queue.empty():

            with start_event_lock:
                if start_event.is_set():
                    inputs = inputs_queue.get()

                    other_start_event.set()
                    start_event.clear()
                else:
                    continue
            
            inputs = _process_inputs(inputs)
            current_inputs = inputs.copy()

            run_id_snapshot = current_inputs.get("run_id", -1)

            inputs = merge_current_and_history(
                global_history[-global_history_limit:],
                inputs,
                skip_history_vision=True,
                move_image_token_to_start=True
            )
        
            if "prompt" in inputs:
                # Process multimodal tokens
                inputs["prompt_token_ids"] = tokenizer_image_audio_token(inputs["prompt"], tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX)
            else:
                assert "prompt_token_ids" in inputs, "Either 'prompt' or 'prompt_token_ids' must be provided."

            print(f"Process {cuda_devices} is processing inputs: {inputs}")

            inputs.pop("prompt", None)

            llm_start_time = time.time()
            output = llm.generate(inputs,
                sampling_params=sampling_params,
            )
            llm_end_time = time.time()
            print(f"{Colors.GREEN}LLM process time: {llm_end_time - llm_start_time}{Colors.RESET}")

            llm_output = output[0].outputs[0].text
            print(f"LLM ouput: {llm_output}")
            # First sentence mark
            llm_output = '$$FIRST_SENTENCE_MARK$$' + llm_output

            async def collect_results(llm_output):
                results = []
                is_first_time_to_work = True
                history_generated_text = ''
                for newly_generated_text in llm_output:
                    # is_negative = judge_negative(newly_generated_text)
                    is_negative = False

                    if not is_negative:
                        history_generated_text += newly_generated_text
                        if is_first_time_to_work:
                            print(f"Process {cuda_devices} is about to interrupt other process")
                            stop_event.clear()
                            other_stop_event.set()
                            clear_queue(outputs_queue)
                            clear_queue(tts_outputs_queue)
                            is_first_time_to_work = False

                        if not stop_event.is_set():
                            results.append(newly_generated_text)
                            history_generated_text = history_generated_text.replace('☞ ', '').replace('☞', '')                            
                            if newly_generated_text in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"]:
                                outputs_queue.put({"id": llm_id, "response": history_generated_text, "run_id": run_id_snapshot},)
                                history_generated_text = ''
                        else:
                            print(f"Process {cuda_devices} is interrupted.")
                            break
                    else:
                        print(f"Process {cuda_devices} is generating negative text.")
                        break
                
                current_inputs["response"] = "".join(results)
                if not current_inputs["response"] == "":
                    global_history.append(current_inputs)
                return results

            results = loop.run_until_complete(collect_results(llm_output))

def tts_worker(
    model_path,
    inputs_queue,
    outputs_queue,
    worker_ready,
    wait_workers_ready,
    record_queue=None,
    run_id=None
):
    import torch
    import torchaudio
    from vita.model.vita_tts.decoder.llm2tts import llm2TTS
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    from transformers import AutoTokenizer
    
    def audio_file_to_html(audio_file: str) -> str:
        """
        Convert audio file to HTML audio player.

        Args:
            audio_file: Path to audio file

        Returns:
            audio_player: HTML audio player that auto-plays
        """
        # Read in audio file to audio_bytes
        audio_bytes = io.BytesIO()
        with open(audio_file, "rb") as f:
            audio_bytes.write(f.read())

        # Generate audio player HTML object for autoplay
        audio_bytes.seek(0)
        audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
        audio_player = (
            f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'
        )
        return audio_player


    def remove_uncommon_punctuation(text):
        common_punctuation = ".,!?;:()[]，。！？、：；（） "
        uncommon_punctuation_pattern = rf"[^\w\s{re.escape(common_punctuation)}]"
        cleaned_text = re.sub(uncommon_punctuation_pattern, "", text)

        return cleaned_text
    
    def remove_special_tokens(input_str):
        # Remove special tokens
        special_tokens = ['☞', '☟', '☜', '<unk>', '<|im_end|>']
        for token in special_tokens:
            input_str = input_str.replace(token, '')
        return input_str

    def replace_equation(sentence):

        special_notations = {
            "sin": " sine ",
            "cos": " cosine ",
            "tan": " tangent ",
            "cot": " cotangent ",
            "sec": " secant ",
            "csc": " cosecant ",
            "log": " logarithm ",
            "exp": "e^",
            "sqrt": "根号 ",
            "abs": "绝对值 ",
        }
        
        special_operators = {
            "+": "加",
            "-": "减",
            "*": "乘",
            "/": "除",
            "=": "等于",
            '!=': '不等于',
            '>': '大于',
            '<': '小于',
            '>=': '大于等于',
            '<=': '小于等于',
        }

        greek_letters = {
            "α": "alpha ",
            "β": "beta ",
            "γ": "gamma ",
            "δ": "delta ",
            "ε": "epsilon ",
            "ζ": "zeta ",
            "η": "eta ",
            "θ": "theta ",
            "ι": "iota ",
            "κ": "kappa ",
            "λ": "lambda ",
            "μ": "mu ",
            "ν": "nu ",
            "ξ": "xi ",
            "ο": "omicron ",
            "π": "派 ",
            "ρ": "rho ",
            "σ": "sigma ",
            "τ": "tau ",
            "υ": "upsilon ",
            "φ": "phi ",
            "χ": "chi ",
            "ψ": "psi ",
            "ω": "omega "
        }

        sentence = sentence.replace('**', ' ')

        sentence = re.sub(r'(?<![\d)])-(\d+)', r'负\1', sentence)

        for key in special_notations:
            sentence = sentence.replace(key, special_notations[key]) 
        for key in special_operators:
            sentence = sentence.replace(key, special_operators[key])
        for key in greek_letters:
            sentence = sentence.replace(key, greek_letters[key])


        sentence = re.sub(r'\(?(\d+)\)?\((\d+)\)', r'\1乘\2', sentence)
        sentence = re.sub(r'\(?(\w+)\)?\^\(?(\w+)\)?', r'\1的\2次方', sentence)
        
        return sentence

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_embedding = load_model_embemding(model_path).to(device)
    tts = llm2TTS(os.path.join(model_path, 'vita_tts_ckpt/'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    worker_ready.set()
    if not isinstance(wait_workers_ready, list):
        wait_workers_ready = [wait_workers_ready]

    past_llm_id = 0

    while True:
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        tts_input_text = ""
        while not inputs_queue.empty():
            time.sleep(0.03)

            stop_at_punc_or_len = False
            response = inputs_queue.get()
            llm_id, newly_generated_text = response["id"], response["response"]
            cur_run_id = response.get("run_id",-1) 

            for character in newly_generated_text:
                
                if  past_llm_id != 0 and past_llm_id != llm_id:
                    tts_input_text = ""
                    outputs_queue.put({"id": llm_id, "response": ("|PAUSE|", None, 0.2)})
                
                tts_input_text += character

                past_llm_id = llm_id
                if character in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"] and len(tts_input_text) >= 5:
                    stop_at_punc_or_len = True
                    break

            if stop_at_punc_or_len:
                break

        if tts_input_text.strip() == "":
            continue

        if '$$FIRST_SENTENCE_MARK$$' in  tts_input_text.strip():
            codec_chunk_size = 20
            seg_threshold = 0.1
            tts_input_text = tts_input_text.replace('$$FIRST_SENTENCE_MARK$$', '').replace('，', '。').replace(',', '。')
            IS_FIRST_SENTENCE = True
        else:
            codec_chunk_size = 40
            seg_threshold = 0.015
            IS_FIRST_SENTENCE = False
        tts_input_text = remove_special_tokens(tts_input_text)
        tts_input_text = replace_equation(tts_input_text)
        tts_input_text = tts_input_text.lower()

        if tts_input_text.strip() == "":
            continue
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = llm_embedding(torch.tensor(tokenizer.encode(tts_input_text)).to(device))
        for seg in tts.run(embeddings.reshape(-1, 896).unsqueeze(0), decoder_topk,
                            None, 
                            codec_chunk_size=codec_chunk_size,
                            codec_padding_size=codec_padding_size,
                            seg_threshold=seg_threshold):

            generation_ts = time.monotonic() 

            if IS_FIRST_SENTENCE:
                try:
                    split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                    seg = seg[:, :, split_idx:]
                except:
                    print('Do not need to split')
                    pass

            seg = torch.cat([seg], -1).float().cpu()
            audio_data = (seg.squeeze().numpy() * 32768.0).astype(np.int16)

            audio_duration = seg.shape[-1]/24000


            if past_llm_id == 0 or past_llm_id == llm_id:
                outputs_queue.put({"id": llm_id, "response": (tts_input_text, audio_data, audio_duration),"run_id": cur_run_id})
                if record_queue is not None:
                    record_queue.put({"id": llm_id, "response": (tts_input_text, audio_data, audio_duration, generation_ts),"run_id": cur_run_id})


def offline_record_sender(sid):
    """
    In offline simulation there is no frontend emit('audio').  
    This thread consumes from TTS_OUTPUT_QUEUE  
    and uses the moment of retrieval as send_ts for aligned writing to disk.
    """
    q = app.config['TTS_OUTPUT_QUEUE']
    print(f"[offline-sender] start sid={sid}")
    while True:
        st = REC_STATE.get(sid)
        if not st or not st.get("running"):
            print(f"[offline-sender] stop sid={sid}")
            break
        try:
            item = q.get(timeout=0.1)
        except Exception:
            continue

        # item ：{"id": llm_id, "response": (tts_input_text, audio_i16, audio_duration)}
        _, audio_i16, _ = item["response"]

        send_ts = time.monotonic()
        _recorder_write_chunk(sid, audio_i16, send_ts)


def recorder_drain_loop(sid):
    q = app.config['TTS_RECORD_QUEUE']
    llm_lock = None  
    print(f"[recorder] drain loop start sid={sid}")
    while True:
        st = REC_STATE.get(sid)
        if not st or not st.get("running"):
            print(f"[recorder] drain loop stop sid={sid}")
            break
        try:
            item = q.get_nowait()
        except Empty:
            time.sleep(0.01)
            continue
        # llm_id, (txt, audio_i16, dur) = item["id"], item["response"]
        llm_id, (txt, audio_i16, dur, generation_ts) = item["id"], item["response"] 

        if "run_id" in item and st.get("run_id") is not None and item["run_id"] != st["run_id"]:
            continue

        cut_ts = st.get("cut_ts", None)
        if cut_ts is not None and generation_ts >= cut_ts:
            continue

        t0 = st.get("t0", 0.0)
        if generation_ts < t0:
            continue
        
        if llm_lock is None:
            llm_lock = llm_id
            st["llm_id"] = llm_id
        if llm_id != llm_lock:
            continue
        # _recorder_write_chunk(sid, audio_i16)
        _recorder_write_chunk(sid, audio_i16, generation_ts) 

def merge_current_and_history(
        global_history,
        current_request,
        skip_history_vision=False,
        move_image_token_to_start=False
    ):

    system_prompts = {
        "video": "<|im_start|>system\nYou are an AI robot and your name is Vita. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the video given by the user, and it is strictly forbidden to answer the question without the content of the video. Please note that you are seeing the video, not the image.<|im_end|>\n",
        "image": "<|im_start|>system\nYou are an AI robot and your name is Vita. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the image given by the user, and it is strictly forbidden to answer the question without the content of the image. Please note that you are seeing the image, not the video.<|im_end|>\n",
        "audio": "<|im_start|>system\nYou are an AI robot and your name is Vita. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- Your answer should not be too long, as simple as possible. <|im_end|>\n"
    }

    def select_system_prompt(current_request):
        if "multi_modal_data" in current_request:
            if "video" in current_request["multi_modal_data"]:
                return system_prompts["video"]
            elif "image" in current_request["multi_modal_data"]:
                return system_prompts["video"]
            elif "audio" in current_request["multi_modal_data"]:
                return system_prompts["audio"]
        return system_prompts["audio"]

    system_prompt = select_system_prompt(current_request)
    # print('current request:',current_request)
    user_prefix = "<|im_start|>user\n"
    bot_prefix = "<|im_start|>assistant\n"
    eos = "<|im_end|>\n"

    if len(global_history) == 0:
        
        current_request["prompt"] = (system_prompt + user_prefix + current_request["prompt"] + eos + bot_prefix).replace('☞ ','☞').replace('☟ ','☟')
        return current_request
    
    # Initialize the current prompt and multimodal data
    current_prompt = system_prompt
    current_multi_modal_data = {"image": [], "audio": [], "video": []}

    # Add the history to the current prompt
    for history in global_history:
        assert "prompt" in history, "Prompt must be provided in history."
        assert "response" in history, "Response must be provided in history."

        if skip_history_vision:
            history_prompt = history["prompt"].replace(IMAGE_TOKEN, "").replace(VIDEO_TOKEN, "")
        else:
            history_prompt = history["prompt"]
        # print('tag1!!!!!!!!!!!!',history_prompt)
        history_prompt = user_prefix + history_prompt + eos + bot_prefix + history["response"] + eos
        for modality in ["image", "audio", "video"]:
            if skip_history_vision and modality in ["image", "video"]:
                continue

            if "multi_modal_data" in history and modality in history["multi_modal_data"]:
                current_multi_modal_data[modality].extend(history["multi_modal_data"][modality])
        current_prompt += history_prompt
    # print('tag2!!!!!!!!!!!!',current_prompt)
    # Add the current request to the current prompt
    current_prompt += user_prefix + current_request["prompt"] + eos + bot_prefix
    for modality in ["image", "audio", "video"]:
        if "multi_modal_data" in current_request and modality in current_request["multi_modal_data"]:
            current_multi_modal_data[modality].extend(current_request["multi_modal_data"][modality])
    # print('tag2!!!!!!!!!!!!',current_prompt)
    for modality in ["image", "audio", "video"]:
        if current_multi_modal_data[modality] == []:
            current_multi_modal_data.pop(modality, None)
    # print('tag3!!!!!!!!!!!!',current_prompt)
    if move_image_token_to_start:
        num_image_tokens = current_prompt.count(IMAGE_TOKEN)
        current_prompt = current_prompt.replace(IMAGE_TOKEN, "")
        current_prompt = current_prompt.replace(system_prompt, "")
        current_prompt = system_prompt + user_prefix + IMAGE_TOKEN * num_image_tokens + current_prompt.replace(user_prefix,'')
    # print('tag4!!!!!!!!!!!!',current_prompt)
    current_request["prompt"] = current_prompt.replace('☞ ','☞').replace('☟ ','☟')
    current_request["multi_modal_data"] = current_multi_modal_data

    return current_request

def send_pcm(sid, request_inputs_queue):
    """
    Sends PCM audio data to the dialogue system for processing.
    """
    chunk_size = connected_users[sid][1].wakeup_and_vad.get_chunk_size()

    # print(f"[[[[VAD DEBUG]]]]chunk_size={chunk_size}, dur_ms={chunk_size/target_sample_rate*1000:.2f} ms")


    print(f"Sid: {sid} Start listening")
    while True:
        if connected_users[sid][1].stop_pcm:
            print(f"Sid: {sid} Stop pcm")
            connected_users[sid][1].stop_generate = True 
            connected_users[sid][1].stop_tts = True
            break
            
        time.sleep(0.01)
        e = connected_users[sid][1].pcm_fifo_queue.get(chunk_size)
        if e is None:
            continue

        res = connected_users[sid][1].wakeup_and_vad.predict(e)
        # print("======vars(connected_users[sid][1].wakeup_and_vad)=========")
        # print(vars(connected_users[sid][1].wakeup_and_vad))

        if res is not None:
            if 'start' in res:
                print(f"Sid: {sid} Vad start")
                mark_interrupt(sid, bump_run=True, clear_record_queue=True)

            elif 'cache_dialog' in res:
                print(f"Sid: {sid} Vad end")

                directory = './chat_history'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                audio_duration = len(res["cache_dialog"]) / target_sample_rate

                if audio_duration < 1:
                    print("The duration of the audio is less than 1s, skipping...")
                    continue

                current_time = datetime.datetime.now()
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                audio_filename = f"{directory}/test_dialog_{timestamp}.wav"
                torchaudio.save(audio_filename, res["cache_dialog"].unsqueeze(0), target_sample_rate)

                video_filename = None
                if len(connected_users[sid][1].collected_images) > 0:
                    video_filename = f"{directory}/test_video_{timestamp}.mp4"
                    save_video(connected_users[sid][1].collected_images, video_filename)

                print("Start to generate response")
                if video_filename:
                    current_request = {
                        "prompt": "<video><audio>",
                        "multi_modal_data": {
                            "video": [video_filename],
                            "audio": [audio_filename],
                        },
                    }
                else:
                    current_request = {
                        "prompt": "<audio>",
                        "multi_modal_data": {
                            "audio": [audio_filename],
                        },
                    }
                print(f"Start to put request into queue {current_request}")

                st = REC_STATE.get(sid)
                if st and "run_id" in st:
                    current_request["run_id"] = st["run_id"]

                request_inputs_queue.put(current_request)

def stream_wav_to_queue(sid, wav_path):
    """
    Split the entire wav file into chunks of the same chunk_size as VAD,  
    and feed them into the pcm queue at a "real-time pace".  
    Afterwards, append ~0.9s of silence to help VAD reliably trigger `ss`  
    instead of being misclassified as `el`/invalid break.
    """
    if sid not in connected_users:
        raise RuntimeError(f"sid {sid} does not exist, please establish a Socket connection first.")


    gp = connected_users[sid][1]
    chunk_size = gp.wakeup_and_vad.get_chunk_size()
    target_sr = 16000

    waveform, sr = torchaudio.load(wav_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform); sr = target_sr

    audio_np = (waveform.squeeze(0).clamp(-1.0, 1.0).numpy() * 32768.0).astype(np.int16)

    # reset 
    gp.interrupt()
    gp.stop_generate = False
    gp.stop_tts = False
    gp.reset()

    print(f"Start streaming simulation: sid={sid}, file={wav_path}, chunk_size={chunk_size}, sr={sr}")

    chunk_dur = chunk_size / float(sr)
    start = time.monotonic()
    n = 0  # Number of chunks already sent

    idx = 0
    while idx < len(audio_np):
        if gp.stop_pcm:
            print(f"Sid: {sid} 模拟中断（stop_pcm=True）")
            break

        # Refresh timeout timer (same Timer as in handle_connect)
        try:
            connected_users[sid][0].cancel()
            connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
            connected_users[sid][0].start()
        except Exception:
            pass

        end = min(idx + chunk_size, len(audio_np))
        chunk = audio_np[idx:end]
        gp.pcm_fifo_queue.put(torch.tensor(chunk, dtype=torch.float32) / 32768.0)
        idx = end
        n += 1

        deadline = start + n * chunk_dur
        now = time.monotonic()
        if deadline > now:
            time.sleep(deadline - now)

    tail = np.zeros(int(sr * 0.9), dtype=np.int16)
    off = 0
    while off < len(tail):
        try:
            connected_users[sid][0].cancel()
            connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
            connected_users[sid][0].start()
        except Exception:
            pass

        seg = tail[off:off+chunk_size]
        gp.pcm_fifo_queue.put(torch.tensor(seg, dtype=torch.float32) / 32768.0)
        off += chunk_size

        n += 1
        deadline = start + n * chunk_dur
        now = time.monotonic()
        if deadline > now:
            time.sleep(deadline - now)

    print(f"sid={sid} wav simulation completed.")



@app.route('/')
def index():
    return render_template('demo.html')

# cleanup at the end to ensure queues are empty before the next run
@app.route('/admin/reset_all', methods=['POST'])
def admin_reset_all():
    try:
        # 1) Stop recording for all sids (so recorder_drain_loop exits cleanly)
        for sid in list(REC_STATE.keys()):
            stop_tts_recording(sid)

        # 2) Clear all queues
        clear_queue(current_app.config['REQUEST_QUEUE'])
        clear_queue(current_app.config['TTS_QUEUE'])
        clear_queue(current_app.config['TTS_OUTPUT_QUEUE'])
        clear_queue(current_app.config['TTS_RECORD_QUEUE'])

        # 3) Clear history (avoid affecting the next run)
        global_history = current_app.config['GLOBAL_HISTORY']
        while len(global_history) > 0:
            global_history.pop()

        # 4) Reset both worker events consistently to ensure the next round starts identically
        current_app.config['WORKER_1_START'].set()
        current_app.config['WORKER_2_START'].clear()

        time.sleep(0.3)

        return jsonify({"ok": True, "msg": "State and queues hard reset"}), 200
    except Exception as e:
        return jsonify({"ok": False, "msg": f"reset_all error: {e}"}), 500


@app.route('/simulate_stream', methods=['POST'])
def simulate_stream():
    """
    Trigger offline simulation: json/form { "wav_path": "...", optional "sid": "...", optional "out_wav": "...", optional "wait": true/false }
    - If sid is not provided and there is exactly one active connection, use that sid automatically.
    - If out_wav is not provided, defaults to ./chat_history/output_{sid}.wav.
    - If wait=true, the API call blocks until the feeding thread finishes (useful for sequential scripts).
    """
    try:
        data = request.get_json() or request.form
        sid = data.get('sid')
        wav_path = data.get('wav_path')
        out_wav = data.get('out_wav')
        wait = bool(str(data.get('wait', 'false')).lower() == 'true')

        if not sid:
            if len(connected_users) == 1:
                sid = list(connected_users.keys())[0]
            else:
                return jsonify({"ok": False, "msg": "sid required (active connections != 1)"}), 800
        if not wav_path or not os.path.exists(wav_path):
            return jsonify({"ok": False, "msg": f"File does not exist or not provided: {wav_path}"}), 400

        if not out_wav:
            out_dir = "./output_wav"
            os.makedirs(out_dir, exist_ok=True)
            out_wav = os.path.join(out_dir, f"output_{sid}.wav")

        app.config['RUN_ID'].value += 1
        cur_run_id = app.config['RUN_ID'].value

        t0 = time.monotonic()
        start_tts_recording(sid, out_wav, t0)

        REC_STATE[sid]["run_id"] = cur_run_id  

         # Start recorder thread 
        threading.Thread(target=recorder_drain_loop, args=(sid,), daemon=True).start()  

        th = threading.Thread(target=stream_wav_to_queue, args=(sid, wav_path), daemon=True)
        th.start()
        if wait:
            th.join()
            wait_and_close_after_drain(sid)

        return jsonify({"ok": True, "sid": sid, "wav_path": wav_path, "out_wav": out_wav, "wait": wait}), 200
    except Exception as e:
        print(f"/simulate_stream error: {e}")
        return jsonify({"ok": False, "msg": f"error: {e}"}), 500

@socketio.on('connect')
def handle_connect():
    if len(connected_users) >= args.max_users:
        print('Too many users connected, disconnecting new user')
        emit('too_many_users')
        return

    sid = request.sid
    connected_users[sid] = []
    connected_users[sid].append(Timer(args.timeout, disconnect_user, [sid]))
    connected_users[sid].append(GlobalParams())
    connected_users[sid][0].start()
    
    request_queue = current_app.config['REQUEST_QUEUE']
    pcm_thread = threading.Thread(target=send_pcm, args=(sid, request_queue,))
    pcm_thread.start()
    print(f'User {sid} connected')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    stop_tts_recording(sid)

    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]
    print(f'User {sid} disconnected')

@socketio.on('recording-started')
def handle_recording_started():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
        mark_interrupt(sid, bump_run=True, clear_record_queue=True)
    else:
        disconnect()
    print('Recording started')

@socketio.on('recording-stopped')
def handle_recording_stopped():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
        mark_interrupt(sid, bump_run=True, clear_record_queue=True)
    else:
        disconnect()
    print('Recording stopped')

@socketio.on('audio')
def handle_audio(data):
    global last_tts_model_id
    sid = request.sid
    if sid in connected_users:
        try:
            if not current_app.config['TTS_OUTPUT_QUEUE'].empty():
                connected_users[sid][0].cancel()
                connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
                connected_users[sid][0].start()

                tts_output_queue = current_app.config['TTS_OUTPUT_QUEUE']
                try:
                    output_data = tts_output_queue.get_nowait()
                    print("output_data", output_data)

                    if output_data is not None:
                        llm_id = output_data["id"]
                        _, audio, length = output_data["response"]

                        print(f"llm_id: {llm_id}, last_tts_model_id: {last_tts_model_id}")
                        if last_tts_model_id != llm_id:
                            print(f"Received output from other process {llm_id}, last output tts model is {last_tts_model_id}, skipping...")
                            socketio.emit('stop_tts', to=sid)
                            mark_interrupt(sid, bump_run=True, clear_record_queue=True)
                        else:
                            print(f"Sid: {sid} Send TTS data")

                            # The real-time moment when speech is sent to the frontend for the first time
                            send_ts = time.monotonic()
                            emit('audio', audio.tobytes())
                            _recorder_write_chunk(sid, audio, send_ts) 

                        last_tts_model_id = llm_id
                except Empty:
                    pass
        
            if connected_users[sid][1].tts_over_time > 0:
                socketio.emit('stop_tts', to=sid)
                connected_users[sid][1].tts_over_time = 0
                        
                mark_interrupt(sid, bump_run=True, clear_record_queue=True)
            
            data = json.loads(data)
            audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
            sample_rate = data['sample_rate']
            
            connected_users[sid][1].pcm_fifo_queue.put(torch.tensor(audio_data, dtype=torch.float32) / 32768.0)

        except Exception as e:
            print(f"Error processing audio: {e}")
    else:
        disconnect()

@socketio.on('video_frame')
def handle_video_frame(data):
    import cv2
    
    sid = request.sid
    if sid in connected_users:
        try:
            image_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            current_time = time.time()
            if current_time - connected_users[sid][1].last_image_time > 1:
                connected_users[sid][1].collected_images.clear()
                print("Clearing the collected images")
            
            connected_users[sid][1].collected_images.append(frame)
            connected_users[sid][1].last_image_time = current_time
            
        except Exception as e:
            print(f"Error processing video frame: {e}")
    else:
        disconnect()

@socketio.on('reset_state')
def handle_reset_state():
    global_history = current_app.config['GLOBAL_HISTORY']
    while len(global_history) > 0:
        global_history.pop()
    print("Resetting the state")

@app.route('/admin/reload_all', methods=['POST'])
def admin_reload_all():
    socketio.emit('force_reload')
    return jsonify({"ok": True, "msg": "broadcasted force_reload"}), 200

@app.route('/admin/list_sids', methods=['GET'])
def admin_list_sids():
    from flask import jsonify
    return jsonify({"sids": list(connected_users.keys())}), 200


def stop_tts_recording(sid):
    st = REC_STATE.get(sid)
    if not st: 
        return
    with st["lock"]:
        st["running"] = False
        try:
            if st.get("fh"):
                st["fh"].close()
                st["fh"] = None
        except Exception:
            pass
    REC_STATE.pop(sid, None)

def start_tts_recording(sid, out_wav_path, t0):
    stop_tts_recording(sid)  
    fh = wave.open(out_wav_path, 'wb')
    fh.setnchannels(1)
    fh.setsampwidth(2)      # int16
    fh.setframerate(TTS_SR)
    REC_STATE[sid] = {"running": True, "t0": t0, "fh": fh, "written": 0, "last_write": t0,"lock": threading.Lock()}
    print(f"[recorder] sid={sid} start -> {out_wav_path}, t0={t0}")

def _recorder_write_chunk(sid, chunk_i16: np.ndarray, send_ts: float):
    """Align writing to the actual playback moment (padding with zeros if necessary).  
        send_ts comes from before emit('audio', ...).
    """
    st = REC_STATE.get(sid)
    if not st or not st["running"]:
        return
        
    if st.get("cut_ts") is not None and send_ts >= st["cut_ts"]:
        return

    with st["lock"]:
        if not st["running"] or "fh" not in st or st["fh"] is None:
            return
        fh = st["fh"]
        written = st["written"]

    expected = int((send_ts - st["t0"]) * TTS_SR)
    gap = expected - written
    if gap > 0:
        fh.writeframes(b"\x00\x00" * gap)
        written += gap

    chunk_bytes = np.asarray(chunk_i16, dtype=np.int16).tobytes()
    fh.writeframes(chunk_bytes)
    written += len(chunk_bytes) // 2
    st["written"] = written
    st["last_write"] = time.monotonic()


def wait_and_close_after_drain(sid, idle_sec=0.8, max_wait_sec=10.0):
    q = app.config['TTS_RECORD_QUEUE']
    start = time.monotonic()
    while time.monotonic() - start < max_wait_sec:
        st = REC_STATE.get(sid)
        if not st or not st.get("running"):
            break
        last_write = st.get("last_write", st["t0"])
        idle = time.monotonic() - last_write
        if q.empty() and idle >= idle_sec:
            break
        time.sleep(0.05)
    stop_tts_recording(sid)


def cleanup_resources():
    """Clean up multiprocessing resources"""
    print("Cleaning up resources...")
    with app.app_context():
        if 'WORKER_1_STOP' in current_app.config:
            current_app.config['WORKER_1_STOP'].set()
        if 'WORKER_2_STOP' in current_app.config:
            current_app.config['WORKER_2_STOP'].set()
        
        if 'REQUEST_QUEUE' in current_app.config:
            clear_queue(current_app.config['REQUEST_QUEUE'])
        if 'TTS_QUEUE' in current_app.config:
            clear_queue(current_app.config['TTS_QUEUE'])
        if 'TTS_OUTPUT_QUEUE' in current_app.config:
            clear_queue(current_app.config['TTS_OUTPUT_QUEUE'])
        
        if 'MODEL_1_PROCESS' in current_app.config:
            current_app.config['MODEL_1_PROCESS'].terminate()
        if 'MODEL_2_PROCESS' in current_app.config:
            current_app.config['MODEL_2_PROCESS'].terminate() 
        if 'TTS_WORKER_PROCESS' in current_app.config:
            current_app.config['TTS_WORKER_PROCESS'].terminate()

atexit.register(cleanup_resources)

if __name__ == "__main__":
    print("Start VITA server")
    
    multiprocessing.set_start_method('spawn', force=True)

    manager = multiprocessing.Manager()

    run_id = manager.Value('i', 0) 

    request_inputs_queue = manager.Queue() 
    tts_inputs_queue = manager.Queue() 
    tts_output_queue = manager.Queue() 
    worker_1_stop_event = manager.Event() 
    worker_2_stop_event = manager.Event() 

    worker_1_start_event = manager.Event() 
    worker_2_start_event = manager.Event()
    worker_1_start_event.set()

    worker_1_2_start_event_lock = manager.Lock()

    llm_worker_1_ready = manager.Event()
    llm_worker_2_ready = manager.Event()

    tts_worker_ready = manager.Event()
    gradio_worker_ready = manager.Event()

    global_history = manager.list()
    global_history_limit = 1

    tts_record_queue = manager.Queue()

    tts_worker_process = multiprocessing.Process(
        target=tts_worker,
        kwargs={
            "model_path": args.model_path,
            "inputs_queue": tts_inputs_queue,
            "outputs_queue": tts_output_queue,
            "worker_ready": tts_worker_ready,
            "wait_workers_ready": [llm_worker_1_ready, llm_worker_2_ready],
            "record_queue": tts_record_queue,
            "run_id": run_id, 
        }
    )

    model_1_process = multiprocessing.Process(
        target=load_model,
        kwargs={
            "llm_id": 1,
            "engine_args": args.model_path, 
            "cuda_devices": "2",
            "inputs_queue": request_inputs_queue,
            "outputs_queue": tts_inputs_queue,
            "tts_outputs_queue": tts_output_queue,
            "start_event": worker_1_start_event,
            "other_start_event": worker_2_start_event,
            "start_event_lock": worker_1_2_start_event_lock,
            "stop_event": worker_1_stop_event,
            "other_stop_event": worker_2_stop_event,
            "worker_ready": llm_worker_1_ready,
            "wait_workers_ready": [llm_worker_2_ready, tts_worker_ready], 
            "global_history": global_history,
            "global_history_limit": global_history_limit,
        }
    )

    model_2_process = multiprocessing.Process(
        target=load_model,
        kwargs={
            "llm_id": 2,
            "engine_args": args.model_path,
            "cuda_devices": "3",
            "inputs_queue": request_inputs_queue,
            "outputs_queue": tts_inputs_queue,
            "tts_outputs_queue": tts_output_queue,
            "start_event": worker_2_start_event,
            "other_start_event": worker_1_start_event,
            "start_event_lock": worker_1_2_start_event_lock,
            "stop_event": worker_2_stop_event,
            "other_stop_event": worker_1_stop_event,
            "worker_ready": llm_worker_2_ready,
            "wait_workers_ready": [llm_worker_1_ready, tts_worker_ready], 
            "global_history": global_history,
            "global_history_limit": global_history_limit,
        }
    )

    model_1_process.start()
    model_2_process.start()
    tts_worker_process.start()

    app.config['REQUEST_QUEUE'] = request_inputs_queue
    app.config['TTS_QUEUE'] = tts_inputs_queue
    app.config['TTS_OUTPUT_QUEUE'] = tts_output_queue
    app.config['WORKER_1_STOP'] = worker_1_stop_event
    app.config['WORKER_2_STOP'] = worker_2_stop_event
    app.config['WORKER_1_START'] = worker_1_start_event
    app.config['WORKER_2_START'] = worker_2_start_event
    app.config['START_LOCK'] = worker_1_2_start_event_lock
    app.config['WORKER_1_READY'] = llm_worker_1_ready
    app.config['WORKER_2_READY'] = llm_worker_2_ready
    app.config['TTS_READY'] = tts_worker_ready
    app.config['GLOBAL_HISTORY'] = global_history
    app.config['MODEL_1_PROCESS'] = model_1_process
    app.config['MODEL_2_PROCESS'] = model_2_process
    app.config['TTS_WORKER_PROCESS'] = tts_worker_process

    app.config['TTS_RECORD_QUEUE'] = tts_record_queue

    app.config['RUN_ID'] = run_id

    import cv2
    import torch
    import torchaudio

    cert_file = "web_demo/vita_html/web/resources/cert.pem"
    key_file = "web_demo/vita_html/web/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    socketio.run(app, host=args.ip, port=args.port, debug=False, ssl_context=(cert_file, key_file))

    model_1_process.join()
    model_2_process.join()
    tts_worker_process.join()
