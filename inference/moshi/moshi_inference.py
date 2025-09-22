import os
import argparse
import torch
import torchaudio
import websocket
import threading
from tqdm import tqdm
import time
import numpy as np
import sphn

# ====== Configurable Parameters ======
TARGET_SAMPLE_RATE = 24000
FRAME_DURATION_MS = 80
FRAME_SIZE = int(TARGET_SAMPLE_RATE * FRAME_DURATION_MS / 1000)
# Send 3 seconds of silence to ensure greeting is fully triggered
PRIMING_SILENCE_SECONDS = 3.0
PRIMING_FRAMES = int(PRIMING_SILENCE_SECONDS * 1000 / FRAME_DURATION_MS)
# Active listening: if no new audio arrives within 0.5s, treat as silence
SILENCE_DETECTION_WINDOW = 0.5 

def get_moshi_output_audio(input_wav_path: str, server_url: str) -> torch.Tensor:
    """
    Perform a full inference cycle for a single evaluation sample,
    including active listening to strip away initial greetings.
    """
    ws = None
    try:
        # 1. Establish connection and handshake
        ws = websocket.create_connection(server_url, timeout=20)
        handshake = ws.recv()
        if not isinstance(handshake, bytes) or handshake != b'\x00':
            print(f"Warning: unexpected handshake signal received: {handshake}")
            
        opus_writer = sphn.OpusStreamWriter(TARGET_SAMPLE_RATE)
        opus_reader = sphn.OpusStreamReader(TARGET_SAMPLE_RATE)
        
        received_pcm_chunks = []
        is_receiving_done = threading.Event()

        def receiver():
            """Receiving thread"""
            try:
                while not is_receiving_done.is_set():
                    message = ws.recv()
                    if not isinstance(message, bytes) or len(message) == 0: continue
                    kind = message[0]
                    if kind == 1:
                        opus_reader.append_bytes(message[1:])
                        pcm_chunk = opus_reader.read_pcm()
                        if pcm_chunk.shape[-1] > 0:
                            received_pcm_chunks.append(torch.from_numpy(pcm_chunk).float())
            except (websocket.WebSocketConnectionClosedException, ConnectionResetError, OSError):
                pass
            finally:
                is_receiving_done.set()

        receiver_thread = threading.Thread(target=receiver, daemon=True)
        receiver_thread.start()
        
        # 2. Priming phase - send silence
        silent_frame_np = np.zeros(FRAME_SIZE, dtype=np.float32)
        for _ in range(PRIMING_FRAMES):
            opus_writer.append_pcm(silent_frame_np)
            opus_bytes = opus_writer.read_bytes()
            if len(opus_bytes) > 0:
                ws.send_bytes(b"\x01" + opus_bytes)
            time.sleep(FRAME_DURATION_MS / 1000.0)
        
        # 3. Active Listening Phase - wait for silence from model
        last_chunk_count = -1
        while True:
            current_chunk_count = len(received_pcm_chunks)
            if current_chunk_count == last_chunk_count:
                # No new audio chunks within the detection window, assume silence
                break
            last_chunk_count = current_chunk_count
            time.sleep(SILENCE_DETECTION_WINDOW)
        
        # 4. Clear buffered greeting audio
        received_pcm_chunks.clear()

        # 5. Inference Phase
        waveform, sample_rate = torchaudio.load(input_wav_path)
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
        waveform_np = waveform.squeeze().numpy()
        for offset in range(0, waveform_np.shape[-1], FRAME_SIZE):
            if not receiver_thread.is_alive(): break
            frame = waveform_np[offset : offset + FRAME_SIZE]
            if frame.shape[-1] < FRAME_SIZE:
                frame = np.pad(frame, (0, FRAME_SIZE - frame.shape[-1]))
            opus_writer.append_pcm(frame)
            opus_bytes = opus_writer.read_bytes()
            if len(opus_bytes) > 0:
                ws.send_bytes(b"\x01" + opus_bytes)
            time.sleep(FRAME_DURATION_MS / 1000.0)

        # 6. Wait and close
        time.sleep(2.0)
        is_receiving_done.set()
        receiver_thread.join(timeout=5)
        ws.close()

        if not received_pcm_chunks:
            return torch.zeros((1, 1))
        
        output_waveform = torch.cat(received_pcm_chunks, dim=-1).unsqueeze(0)
        return output_waveform

    except Exception as e:
        print(f"\n[Error] Failed to process file {os.path.basename(input_wav_path)}: {e}")
        if ws: ws.close()
        return torch.zeros((1, 1))

def run_inference_on_dataset(data_dir: str, moshi_url: str):
    input_files = sorted([os.path.join(root, "input.wav") for root, _, files in os.walk(data_dir) if "input.wav" in files])

    if not input_files:
        print(f"Error: No 'input.wav' files found under directory '{data_dir}'.")
        return
    print(f"Found {len(input_files)} evaluation samples under '{data_dir}'.")

    for input_path in tqdm(input_files, desc="[Step 1] Generating model outputs"):
        output_path = input_path.replace("input.wav", "output.wav")
        output_waveform = get_moshi_output_audio(input_path, moshi_url)
        if output_waveform.numel() > 1:
            torchaudio.save(output_path, output_waveform, TARGET_SAMPLE_RATE)
        else:
            torchaudio.save(output_path, torch.zeros((1, 1)), TARGET_SAMPLE_RATE)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Step 1: Run Moshi inference to generate output.wav files for evaluation dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of evaluation dataset.")
    parser.add_argument("--moshi_url", type=str, default="ws://localhost:8998/api/chat", help="WebSocket URL of a running Moshi server.")
    args = parser.parse_args()

    print("Attempting to connect to Moshi service...")

    try:
        websocket.create_connection(args.moshi_url, timeout=5).close()
        print(" Moshi service connection successful!")
    except Exception as e:
        print(f"Failed to connect to Moshi service: {e}")
        print("Please make sure you have started the server in another terminal with 'python -m moshi.server ...'.")
        exit(1)

    run_inference_on_dataset(args.data_dir, args.moshi_url)
    
    print("\n[Step 1 Completed]")
    print(f"All 'output.wav' files have been generated or updated under '{args.data_dir}'.")
    print("\n[Next Step]")
    print("Please run Step 2 ASR script to generate 'output.json' files for final evaluation.")
