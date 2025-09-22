from __future__ import print_function

import argparse
import os
import json
import queue
import torch
import yaml
import threading
import struct
import time
import torchaudio
import datetime
import builtins
import wave

import numpy as np

from copy import deepcopy
from threading import Timer
from flask import Flask, render_template, request
from flask import jsonify
from flask_socketio import SocketIO, disconnect, emit

from web.parms import GlobalParams
from web.pool import TTSObjectPool, pipelineObjectPool
from web.pem import generate_self_signed_cert


def get_args():
    parser = argparse.ArgumentParser(description='Freeze-Omni')
    parser.add_argument('--model_path', required=True, help='model_path to load')
    parser.add_argument('--llm_path', required=True, help='llm_path to load')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--ip', required=True, help='ip of server')
    parser.add_argument('--port', required=True, help='port of server')
    parser.add_argument('--max_users', type=int, default=5)
    parser.add_argument('--llm_exec_nums', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=6000)
    args = parser.parse_args()
    print(args)
    return args

def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    original_print(f'[{current_time}]', *args, **kwargs)

# init parms
configs = get_args()
# read server related config
server_configs = json.loads(open(configs.model_path + "/server.json").read())
# max users to connect
MAX_USERS = configs.max_users
# number of inference pipelines to use
PIPELINE_NUMS = configs.llm_exec_nums
# timeout to each user
TIMEOUT = configs.timeout

# change print function to add time stamp
original_print = builtins.print
builtins.print = custom_print

# init inference pipelines pool
pipeline_pool = pipelineObjectPool(size=PIPELINE_NUMS, configs=configs)
# inint speech decoder pool
tts_pool = TTSObjectPool(size=MAX_USERS, model_path=configs.model_path)

# init flask app
app = Flask(__name__, template_folder='../web/resources')
socketio = SocketIO(app)
# init connected users
connected_users = {}

def decoder(cur_hidden_state, cur_text, outputs, connected_users, sid, generate_num, last_text, is_last_chunk=False):
    """
    Decodes the current hidden state and text to generate audio segments using speech decoder.

    Parameters:
    - cur_hidden_state (list of torch.Tensor): The current hidden state of the language model.
    - cur_text (str): The current text to be synthesized.
    - connected_users (dict): A dictionary containing information about connected users.
    - sid (str): The session ID of the user.
    - is_last_chunk (bool, optional): Indicates if the current text is the last chunk of the input

    Returns:
    - int: The updated number of audio segments generated.
    """
    hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
    cur_text_procced = connected_users[sid][1].pipeline_obj.pipeline_proc.post_process(cur_text)
    print("Synthesis: ", [cur_text_procced])
    embeddings = connected_users[sid][1].pipeline_obj.pipeline_proc.model.llm_decoder.model.embed_tokens(
                    torch.tensor(connected_users[sid][1].pipeline_obj.pipeline_proc.model.tokenizer.encode(
                        cur_text_procced
                        )).cuda()
                    )
    codec_chunk_size = server_configs['decoder_first_chunk_size']
    codec_padding_size = server_configs['decoder_chunk_overlap_size']
    seg_threshold = server_configs['decoder_seg_threshold_first_pack']
    if generate_num != 0:
        codec_chunk_size = server_configs['decoder_chunk_size']
        seg_threshold = server_configs['decoder_seg_threshold']
    for seg in connected_users[sid][1].tts_obj.tts_proc.run(embeddings.reshape(-1, 896).unsqueeze(0), 
                                                            server_configs['decoder_top_k'], 
                                                            hidden_state_output.reshape(-1, 896).unsqueeze(0), 
                                                            codec_chunk_size, 
                                                            codec_padding_size,
                                                            server_configs['decoder_penalty_window_size'], 
                                                            server_configs['decoder_penalty'],
                                                            server_configs['decoder_N'], 
                                                            seg_threshold):
        if generate_num == 0:
            try:
                split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                seg = seg[:, :, split_idx:]
            except:
                print("Do not need to split")
                pass
        generate_num += 1
        if connected_users[sid][1].tts_over:
            connected_users[sid][1].tts_data.clear()
            connected_users[sid][1].whole_text = ""
            break
        connected_users[sid][1].tts_data.put((seg.squeeze().float().cpu().numpy() * 32768).astype(np.int16))
    return generate_num

def generate(outputs, sid):
    """
    Generates speech dialogue output based on the current state and user session ID.

    Parameters:
    - outputs (dict): A dictionary containing the current state of the dialogue system.
    - sid (str): The session ID of the user.

    Returns:
    - None
    """
    # Stage3: start speak
    connected_users[sid][1].is_generate = True

    outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(None, **outputs)
    connected_users[sid][1].generate_outputs = deepcopy(outputs)

    cur_hidden_state = []
    cur_hidden_state.append(outputs['hidden_state'])

    connected_users[sid][1].whole_text = ""
    # Stage4: contiune speak until stat is set to 'sl'
    # use 'stop' to interrupt generation, stat need to be manually set as 'sl'  
    stop = False
    cur_text = ''
    last_text = ''
    generate_num = 0
    while True:
        if connected_users[sid][1].stop_generate:
            break
        if len(outputs['past_tokens']) > 500:
            stop = True
        if stop:
            break
        del outputs['text']
        del outputs['hidden_state']
        outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(None, **outputs)
        connected_users[sid][1].generate_outputs = deepcopy(outputs)
        if outputs['stat'] == 'cs':
            cur_hidden_state.append(outputs['hidden_state'])
            if "�" in outputs['text'][len(last_text):]:
                continue
            connected_users[sid][1].whole_text += outputs['text'][len(last_text):]
            cur_text += outputs['text'][len(last_text):]
            # print([connected_users[sid][1].whole_text])
            if generate_num == 0 or (len(cur_hidden_state) >= 20):
                suffix_list = [",", "，", "。", "：", "？", "！", ".", ":", "?","!", "\n"]
            else:
                suffix_list = ["。", "：", "？", "！", ".", "?","!", "\n"]
            if outputs['text'][len(last_text):].endswith(tuple(suffix_list)) and (len(cur_hidden_state) >= 4):
                if outputs['text'][len(last_text):].endswith(".") and last_text[-1].isdigit():
                    pass
                else:
                    if not connected_users[sid][1].tts_over:
                        if len(cur_hidden_state) > 0:
                            generate_num = decoder(cur_hidden_state, 
                                                   cur_text, outputs, 
                                                   connected_users, 
                                                   sid, 
                                                   generate_num, 
                                                   last_text)
                            cur_text = ""
                            cur_hidden_state = []
            last_text = outputs['text']
        else:
            break
    if not connected_users[sid][1].tts_over:
        if len(cur_hidden_state) != 0:
            generate_num = decoder(cur_hidden_state, 
                                   cur_text, outputs, 
                                   connected_users, 
                                   sid, 
                                   generate_num, 
                                   last_text, 
                                   is_last_chunk=True)
            cur_text = ""
    connected_users[sid][1].is_generate = False

def llm_prefill(data, outputs, sid, is_first_pack=False):
    """
    Prefill stage: handle sl / cl / el based on VAD state machine.

    Key change:
      When outputs['stat'] == 'el' (invalid break), also call interrupt() to clear caches
      so we won't get stuck in an invalid endpoint and fail to enter 'ss' afterward.
    """
    if data['status'] == 'sl':
        outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
            torch.tensor(data['feature']), **outputs)

    if data['status'] == 'el':
        connected_users[sid][1].wakeup_and_vad.in_dialog = False
        connected_users[sid][1].interrupt()
        print("Sid: ", sid, " Detect vad time out")

    if data['status'] == 'cl':
        if outputs['stat'] == 'cl':
            outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
                torch.tensor(data['feature']), **outputs)
        if is_first_pack:
            outputs['stat'] = 'cl'

        if outputs['stat'] == 'el':
            connected_users[sid][1].wakeup_and_vad.in_dialog = False
            connected_users[sid][1].interrupt()  # clear cache on invalid break
            print("Sid: ", sid, " Detect invalid break")

        if outputs['stat'] == 'ss':
            connected_users[sid][1].interrupt()
            print("Sid: ", sid, " Detect break")
            connected_users[sid][1].wakeup_and_vad.in_dialog = False
            generate_thread = threading.Thread(target=generate, args=(deepcopy(outputs), sid))
            generate_thread.start()
    return outputs


def send_pcm(sid):
    """
    Sends PCM audio data to the dialogue system for processing.

    Parameters:
    - sid (str): The session ID of the user.
    """

    chunk_szie = connected_users[sid][1].wakeup_and_vad.get_chunk_size()

    print("Sid: ", sid, " Start listening")
    while True:
        if connected_users[sid][1].stop_pcm:
            print("Sid: ", sid, " Stop pcm")
            connected_users[sid][1].stop_generate = True
            connected_users[sid][1].stop_tts = True
            break
        time.sleep(0.01)
        e = connected_users[sid][1].pcm_fifo_queue.get(chunk_szie)
        if e is None:
            continue
        print("Sid: ", sid, " Received PCM data: ", len(e))
        res = connected_users[sid][1].wakeup_and_vad.predict(np.float32(e))
        # print("VAD result:", res)
        if res['status'] == 'sl':
            print("Sid: ", sid, " Vad start")
            outputs = deepcopy(connected_users[sid][1].generate_outputs)
            outputs['adapter_cache'] = None
            outputs['encoder_cache'] = None
            outputs['pe_index'] = 0
            outputs['stat'] = 'sl'
            outputs['last_id'] = None
            if 'text' in outputs:
                del outputs['text']
            if 'hidden_state' in outputs:
                del outputs['hidden_state']

            send_dict = {}
            for i in range(len(res['feature_last_chunk'])):
                if i == 0:
                    send_dict['status'] = 'sl'
                else:
                    send_dict['status'] = 'cl'
                send_dict['feature'] = res['feature_last_chunk'][i]
                outputs = llm_prefill(send_dict, outputs, sid, is_first_pack=True)
            send_dict['status'] = 'cl'
            send_dict['feature'] = res['feature']
            outputs = llm_prefill(send_dict, outputs, sid)

        elif res['status'] == 'cl' or res['status'] == 'el':
            send_dict = {}
            send_dict['status'] = res['status']
            send_dict['feature'] = res['feature']
            outputs = llm_prefill(send_dict, outputs, sid)

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

def stream_wav_to_queue(sid, wav_path):
    """
    Stream a full WAV file into the PCM queue using the same VAD chunk size, 
    feeding in "real-time pace".

    Key changes:
      - Refresh the session Timer (keepalive) on each chunk to avoid TIMEOUT.
      - Append ~0.9s of trailing silence to reliably trigger 'ss' instead of 'el'/invalid break.
    """
    if sid not in connected_users:
        raise RuntimeError(f"sid {sid} does not exist; please establish the Socket connection first.")

    gp = connected_users[sid][1]
    chunk_size = gp.wakeup_and_vad.get_chunk_size()
    target_sr = 16000

    # Load and resample to 16k mono
    waveform, sr = torchaudio.load(wav_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform); sr = target_sr

    # Convert to int16, then normalized float32 when enqueueing (matching handle_audio)
    audio_np = (waveform.squeeze(0).clamp(-1.0, 1.0).numpy() * 32768.0).astype(np.int16)

    # Reset states
    gp.interrupt()
    gp.stop_generate = False
    gp.stop_tts = False
    gp.reset()

    sleep_sec = chunk_size / float(sr)
    print(f"Start simulated streaming: sid={sid}, file={wav_path}, chunk_size={chunk_size}, sr={sr}")

    # ---- feeding ----
    idx = 0
    while idx < len(audio_np):
        if gp.stop_pcm:
            print(f"Sid: {sid} simulated stop (stop_pcm=True)")
            break

        # Refresh timeout timer (keepalive)
        try:
            connected_users[sid][0].cancel()
            connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
            connected_users[sid][0].start()
        except Exception:
            pass

        end = min(idx + chunk_size, len(audio_np))
        chunk = audio_np[idx:end]
        connected_users[sid][1].pcm_fifo_queue.put(
            torch.tensor(chunk, dtype=torch.float32) / 32768.0
        )
        idx = end
        time.sleep(sleep_sec)  # real-time pace

    # ---- append ~0.9s trailing silence to help endpoint detection (ss) ----
    tail = np.zeros(int(sr * 0.9), dtype=np.int16)
    off = 0
    while off < len(tail):
        # Keepalive
        try:
            connected_users[sid][0].cancel()
            connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
            connected_users[sid][0].start()
        except Exception:
            pass

        seg = tail[off:off+chunk_size]
        connected_users[sid][1].pcm_fifo_queue.put(
            torch.tensor(seg, dtype=torch.float32) / 32768.0
        )
        off += chunk_size
        time.sleep(sleep_sec)

    print(f"WAV simulation finished for sid={sid}.")


TTS_SR = 24000
# Maintain per-sid recording state: {sid: {"running": bool, "t0": float, "fh": wave_handle, "written": int, "thread": Thread}}
REC_STATE = {}

def _recorder_loop(sid):
    """Background thread: read chunks from connected_users[sid][1].tts_data and write to WAV aligned to timeline."""
    st = REC_STATE.get(sid)
    if not st: 
        return
    t0 = st["t0"]; fh = st["fh"]
    written = st["written"]

    while st["running"]:
        # Session may have been disconnected
        if sid not in connected_users:
            break

        q = connected_users[sid][1].tts_data
        # Get one TTS chunk (np.int16); wait briefly if empty
        if hasattr(q, "is_empty") and q.is_empty():
            time.sleep(0.01)
            continue
        try:
            chunk_i16 = q.get()
        except Exception:
            time.sleep(0.01)
            continue
        if chunk_i16 is None:
            break

        # Target timeline (since the moment input.wav feeding started)
        now = time.monotonic()
        expected = int((now - t0) * TTS_SR)
        gap = expected - written
        if gap > 0:
            fh.writeframes(b"\x00\x00" * gap)
            written += gap

        # Write current TTS chunk (ensure int16)
        chunk_bytes = np.asarray(chunk_i16, dtype=np.int16).tobytes()
        fh.writeframes(chunk_bytes)
        written += (len(chunk_bytes) // 2)

        st["written"] = written

    try:
        if fh: fh.close()
    finally:
        print(f"[recorder] sid={sid} stop, written_samples={written}")

def stop_tts_recording(sid):
    """Stop recorder and clean up."""
    st = REC_STATE.get(sid)
    if not st: 
        return
    st["running"] = False
    th = st.get("thread")
    if th and th.is_alive():
        th.join(timeout=2)
    try:
        if st.get("fh"):
            st["fh"].close()
    except Exception:
        pass
    REC_STATE.pop(sid, None)

def start_tts_recording(sid, out_wav_path, t0):
    """Start recorder (t0 = the time when input.wav feeding begins)."""

    # Stop any existing recorder for this sid
    stop_tts_recording(sid)

    fh = wave.open(out_wav_path, 'wb')
    fh.setnchannels(1)
    fh.setsampwidth(2)      # int16
    fh.setframerate(TTS_SR)

    REC_STATE[sid] = {
        "running": True,
        "t0": t0,
        "fh": fh,
        "written": 0,
        "thread": None,
    }
    th = threading.Thread(target=_recorder_loop, args=(sid,), daemon=True)
    REC_STATE[sid]["thread"] = th
    th.start()
    print(f"[recorder] sid={sid} start -> {out_wav_path}, t0={t0}")

    
@app.route('/simulate_stream', methods=['POST'])
def simulate_stream():
    """
    Trigger offline simulation: json/form { "wav_path": "...", optional "sid": "...", optional "out_wav": "...", optional "wait": true/false }
    - If no sid is provided and there is exactly one connection, use that sid automatically; otherwise return 800 (consistent with previous logic).
    - out_wav can be customized; if omitted, defaults to output_{sid}.wav.
    - When wait=true, the endpoint waits for the feeding thread to finish before returning (convenient for scripting).
    """
    try:
        data = request.get_json() or request.form
        sid = data.get('sid')
        wav_path = data.get('wav_path')
        out_wav = data.get('out_wav')
        wait = bool(str(data.get('wait', 'false')).lower() == 'true')

        # choose sid
        if not sid:
            if len(connected_users) == 1:
                sid = list(connected_users.keys())[0]
            else:
                return jsonify({"ok": False, "msg": "sid is required (current connection count != 1)"}), 800

        if not wav_path:
            return jsonify({"ok": False, "msg": "missing wav_path"}), 800
        if not os.path.exists(wav_path):
            return jsonify({"ok": False, "msg": f"file does not exist: {wav_path}"}), 400

        # default output file
        if not out_wav:
            out_dir = "/your/path/Freeze-Omni/output_wav"
            os.makedirs(out_dir, exist_ok=True)
            out_wav = os.path.join(out_dir, f"output_{sid}.wav")

        # start recorder (based on current monotonic clock)
        t0 = time.monotonic()
        start_tts_recording(sid, out_wav, t0)

        # background feeding
        th = threading.Thread(target=stream_wav_to_queue, args=(sid, wav_path), daemon=True)
        th.start()
        if wait:
            th.join()

        return jsonify({"ok": True, "sid": sid, "wav_path": wav_path, "out_wav": out_wav, "wait": wait}), 200
    except Exception as e:
        print(f"/simulate_stream error: {e}")
        return jsonify({"ok": False, "msg": f"error: {e}"}), 500


@app.route('/')
def index():
    return render_template('demo.html')

@socketio.on('connect')
def handle_connect():
    if len(connected_users) >= MAX_USERS:
        print('Too many users connected, disconnecting new user')
        emit('too_many_users')
        return

    sid = request.sid
    connected_users[sid] = []
    connected_users[sid].append(Timer(TIMEOUT, disconnect_user, [sid]))
    connected_users[sid].append(GlobalParams(tts_pool, pipeline_pool))
    connected_users[sid][0].start()
    connected_users[sid][1].set_prompt(
        "The answers should be as clear and concise as possible."
    )
    pcm_thread = threading.Thread(target=send_pcm, args=(sid,))
    pcm_thread.start()
    tts_pool.print_info()
    pipeline_pool.print_info()
    print(f'User {sid} connected')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    # stop TTS recording for this user (if /simulate_stream is running)
    stop_tts_recording(sid)  
    if sid in connected_users:
        try:
            # cancel timeout timer
            connected_users[sid][0].cancel()
        except Exception:
            pass

        # terminate background processing within the session
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].stop_generate = True
        connected_users[sid][1].stop_tts = True

        # release inference/decoder resources
        connected_users[sid][1].release()
        time.sleep(0.1)

        # remove from session table
        del connected_users[sid]
    print(f'User {sid} disconnected (client)')

@socketio.on('recording-started')
def handle_recording_started():
    sid = request.sid
    if sid in connected_users:
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    print('Recording started')

@socketio.on('recording-stopped')
def handle_recording_stopped():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    print('Recording stopped')

@socketio.on('prompt_text')
def handle_prompt_text(text):
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][1].set_prompt(text)
        print("Sid: ", sid, "Prompt set as: ", text)
        socketio.emit('prompt_success', to=sid)
    else:
        disconnect()

@socketio.on('audio')
def handle_audio(data):
    sid = request.sid
    if sid in connected_users:
        if not connected_users[sid][1].tts_data.is_empty():
            connected_users[sid][0].cancel()
            connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
            connected_users[sid][0].start()
            output_data = connected_users[sid][1].tts_data.get()
            if output_data is not None:
                print("Sid: ", sid, "Send TTS data")
                emit('audio', output_data.astype(np.int16).tobytes())

        if connected_users[sid][1].tts_over_time > 0:
            socketio.emit('stop_tts', to=sid)
            connected_users[sid][1].tts_over_time = 0
        
        data = json.loads(data)
        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        sample_rate = data['sample_rate']
        
        connected_users[sid][1].pcm_fifo_queue.put(audio_data.astype(np.float32) / 32768.0)

    else:
        disconnect()

@app.route('/admin/reload_all', methods=['POST'])
def admin_reload_all():
    socketio.emit('force_reload')
    return jsonify({"ok": True, "msg": "broadcasted force_reload"}), 200

@app.route('/admin/list_sids', methods=['GET'])
def admin_list_sids():
    from flask import jsonify
    return jsonify({"sids": list(connected_users.keys())}), 200


if __name__ == "__main__":
    print("Start Freeze-Omni sever") 
    cert_file = "/your/path/Freeze-Omni/web/resources/cert.pem"
    key_file = "/your/path/Freeze-Omni/web/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    socketio.run(app, host=configs.ip, port=configs.port, ssl_context=(cert_file, key_file))