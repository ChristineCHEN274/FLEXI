import os
import time
import wave
import asyncio
import argparse
import traceback
import audioop
import glob

import pyaudio
from google import genai
from google.genai import types

# ===== Audio Config =====
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# After audio streaming ends, wait for this long to ensure the final reply is received and played.
# If the responses are often long, you may increase this value.
SETTLE_TIME_SECONDS = 5.0

# ===== Model Config =====
MODEL = "models/gemini-2.5-flash-live-preview"
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key="YOUR_API_KEY"
)

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self, wav_path: str, save_aligned_path: str):
        self.session = None
        self.wav_path = wav_path
        self.save_aligned_path = save_aligned_path
        self.play_q = asyncio.Queue()
        self.t0 = None

    async def stream_wav_file(self):
        # 此函数与之前的版本相同
        print(f"Streaming WAV file: {self.wav_path}")
        with wave.open(self.wav_path, "rb") as wf:
            in_channels, in_sampwidth, in_rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            if in_channels not in (1, 2): raise (f"Only mono or stereo supported, detected {in_channels}.")
            in_frames_per_iter, ratecv_state = max(1, int(CHUNK_SIZE * in_rate / SEND_SAMPLE_RATE)), None
            first_input_sent = False
            while True:
                raw = wf.readframes(in_frames_per_iter)
                if not raw: break
                if in_sampwidth != 2: raw = audioop.lin2lin(raw, in_sampwidth, 2)
                if in_channels == 2: raw = audioop.tomono(raw, 2, 0.5, 0.5)
                if in_rate != SEND_SAMPLE_RATE: raw, ratecv_state = audioop.ratecv(raw, 2, 1, in_rate, SEND_SAMPLE_RATE, ratecv_state)
                target_bytes, offset, n = CHUNK_SIZE * 2, 0, len(raw)
                while offset < n:
                    chunk = raw[offset: offset + target_bytes]
                    offset += len(chunk)
                    if not chunk: break
                    if not first_input_sent:
                        self.t0 = time.monotonic()
                        self._next_t = self.t0
                        first_input_sent = True
                    await self.session.send(input={"data": chunk, "mime_type": "audio/pcm"})
                    self._next_t += CHUNK_SIZE / SEND_SAMPLE_RATE
                    sleep_s = self._next_t - time.monotonic()
                    if sleep_s > 0: await asyncio.sleep(sleep_s)
                    else: self._next_t = time.monotonic()
        print("Finished streaming file.")

    async def receive_audio(self):
        print("Ready to receive server audio responses...")
        while True:
            try:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        self.play_q.put_nowait(data)
                    if text := response.text:
                        print(text, end="", flush=True)
                
                # At the end of a turn, clear any leftover queue items
                cleared = 0
                while not self.play_q.empty():
                    try:
                        self.play_q.get_nowait(); cleared += 1
                    except asyncio.QueueEmpty: break
                if cleared > 1: 
                    print(f"\n[interrupt] Cleared {cleared} chunks from playback queue.")
                print()

            except asyncio.CancelledError:
                 # Exit cleanly when task is cancelled
                break
            except Exception as e:
                s = str(e).lower()
                if "1007" in s or "invalid frame payload data" in s or "request contains an invalid argument" in s:
                    # silently ignore these common benign errors
                    break
                else:
                    print(f"\nError while receiving audio: {e}")
                    break

    async def play_audio(self):
        output_stream = await asyncio.to_thread(pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True, frames_per_buffer=CHUNK_SIZE)
        print("Audio player ready.")
        wf_aligned = None
        if self.save_aligned_path:
            os.makedirs(os.path.dirname(self.save_aligned_path), exist_ok=True)
            wf_aligned = wave.open(self.save_aligned_path, "wb")
            wf_aligned.setnchannels(CHANNELS); wf_aligned.setsampwidth(pya.get_sample_size(FORMAT)); wf_aligned.setframerate(RECEIVE_SAMPLE_RATE)
        aligned_written_samples = 0
        try:
            while True:
                chunk = await self.play_q.get()
                t_play = time.monotonic()
                await asyncio.to_thread(output_stream.write, chunk)
                if wf_aligned:
                    if self.t0 is None: self.t0 = t_play
                    ideal_samples = int(round((t_play - self.t0) * RECEIVE_SAMPLE_RATE))
                    if ideal_samples > aligned_written_samples:
                        gap = ideal_samples - aligned_written_samples
                        wf_aligned.writeframes(b"\x00\x00" * gap)
                        aligned_written_samples += gap
                    wf_aligned.writeframes(chunk)
                    aligned_written_samples += len(chunk) // 2
        except asyncio.CancelledError:
            pass
        finally:
            if wf_aligned: wf_aligned.close(); print(f"Aligned audio saved to: {self.save_aligned_path}")
            if output_stream: output_stream.stop_stream(); output_stream.close()
            print("Audio player closed.")

    async def run(self):
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                
                async with asyncio.TaskGroup() as tg:
                    # 1. Start persistent receive and play tasks
                    receive_task = tg.create_task(self.receive_audio())
                    play_task = tg.create_task(self.play_audio())

                    # 2. Start the controller task (stream and shutdown)
                    await tg.create_task(self._stream_and_shutdown(receive_task, play_task))

        except asyncio.CancelledError:
            print("Run method was externally cancelled.")
        except Exception as e:
            print(f"Top-level error while processing file {self.wav_path}: {e}")
            traceback.print_exc()

    async def _stream_and_shutdown(self, receive_task, play_task):
        """Internal controller: streams audio, then terminates other tasks"""
        try:
            # Stream WAV file
            await self.stream_wav_file()
            
            # Notify server of end of turn
            await self.session.send(end_of_turn=True)

            # Wait for final responses
            print(f"Streaming finished, waiting {SETTLE_TIME_SECONDS} seconds for final responses...")
            await asyncio.sleep(SETTLE_TIME_SECONDS)

        finally:
            print("Settle time finished, shutting down tasks for this file...")
            if not receive_task.done():
                receive_task.cancel()
            if not play_task.done():
                play_task.cancel()


# ===== Batch processing =====
def parse_args():
    parser = argparse.ArgumentParser(description="Batch process WAV files and interact with Gemini API via speech.")
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory containing subfolders with input.wav files.")
    parser.add_argument("--output-filename", type=str, default="output.wav", help="Output filename saved in each subfolder.")
    return parser.parse_args()

async def batch_process(base_dir: str, output_filename: str):
    search_pattern = os.path.join(base_dir, '*', 'input.wav')
    wav_files = glob.glob(search_pattern)

    if not wav_files:
        print(f"No 'input.wav' files found under pattern '{search_pattern}'.")
        return
    print(f"Found {len(wav_files)} 'input.wav' files. Starting processing...")
    
    for i, input_path in enumerate(wav_files):
        print(f"\n{'='*20} File {i+1}/{len(wav_files)} {'='*20}")
        print(f"Processing: {input_path}")
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, output_filename)
        print(f"Output path: {output_path}")
        audio_processor = AudioLoop(wav_path=input_path, save_aligned_path=output_path)
        await audio_processor.run()
        print(f"Finished processing {os.path.basename(input_path)}.")
        print(f"{'='*50}\n")
    print("All files processed!")

if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(batch_process(args.base_dir, args.output_filename))
    except KeyboardInterrupt:
        print("\nUser interrupted program.")
    finally:

        pya.terminate()
