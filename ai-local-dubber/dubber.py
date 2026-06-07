#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import subprocess
import requests
import torch
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

def setup_environment(workspace):
    """Creates a clean workspace folder, clearing out old temp files."""
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace, exist_ok=True)

def run_phase_1_demucs(input_video, workspace):
    print("[+] Phase 1: Extracting vocals and background track via Demucs...")
    raw_audio = os.path.join(workspace, "raw_audio.wav")
    
    # 1. Extract audio track from video
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", raw_audio
    ], check=True)
    
    print("    -> Intercepting TorchAudio to bypass ABI bug and executing Demucs...")
    
    # 2. Dynamically generate the Monkey-Patch wrapper script
    wrapper_path = os.path.join(workspace, "demucs_wrapper.py")
    with open(wrapper_path, "w") as f:
        f.write("""import sys
import torch
import torchaudio
import soundfile as sf

# Monkey-patch TorchAudio to completely bypass TorchCodec
def safe_save(filepath, src, sample_rate, **kwargs):
    # Demucs outputs (channels, frames) but Soundfile expects (frames, channels)
    audio_np = src.T.cpu().numpy()
    sf.write(str(filepath), audio_np, sample_rate)

def safe_load(filepath, **kwargs):
    wav, sr = sf.read(str(filepath))
    if wav.ndim == 1:
        wav = wav.reshape(-1, 1)
    tensor = torch.from_numpy(wav.T).float()
    return tensor, sr

# Force Torchaudio to use our safe, raw implementations
torchaudio.save = safe_save
torchaudio.load = safe_load

# Launch Demucs internally within the patched environment
from demucs.separate import main
sys.argv[0] = 'demucs'
main()
""")

    # 3. Execute the wrapper instead of the standard Demucs CLI
    subprocess.run([
        sys.executable, wrapper_path, 
        "--two-stems", "vocals", 
        "-o", workspace, 
        raw_audio
    ], check=True)
    
    # Locate Demucs outputs safely on Windows
    demucs_out_dir = os.path.join(workspace, "htdemucs", "raw_audio")
    return (os.path.join(demucs_out_dir, "vocals.wav"), 
            os.path.join(demucs_out_dir, "no_vocals.wav"))

def run_phase_2_transcribe(vocals_path):
    print("[+] Phase 2: Transcribing original English audio via Faster-Whisper...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(vocals_path, beam_size=5)
    
    chunks = []
    for segment in segments:
        chunks.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
    
    del model
    torch.cuda.empty_cache() # Flush VRAM immediately
    return chunks

def run_phase_3_translate(chunks, target_lang):
    print(f"[+] Phase 3: Translating dialogue via Local Ollama (Llama-3.1)...")
    url = "http://localhost:11434/api/generate"
    
    for chunk in chunks:
        payload = {
            "model": "llama3.1:8b",
            "prompt": f"Text to translate: '{chunk['text']}'",
            # The STRICT System Prompt ensures no English leaks through
            "system": f"You are a strict translation API. Translate the provided text into {target_lang} script. Output ONLY the {target_lang} script. Do not output English words. Do not include quotes, notes, or conversational filler like 'Here is the translation'.",
            "stream": False
        }
        try:
            r = requests.post(url, json=payload).json()
            # Strip any accidental spaces, quotes, or newlines Llama tries to add
            chunk["translated_text"] = r["response"].strip(' \'"\n')
        except Exception as e:
            chunk["translated_text"] = chunk["text"]
            
    return chunks

def run_phase_4_synthesis(chunks, ref_voice, workspace):
    print("[+] Phase 4: Generating cloned voice segments via IndicF5...")
    import torch
    import torchaudio
    from transformers import AutoModel
    
    # [THE WINDOWS FFMPEG/TORCHCODEC FIX]
    # Completely bypass torchaudio's broken native decoder by forcing it to use soundfile
    def soundfile_load_fallback(filepath, **kwargs):
        data, sr = sf.read(filepath, dtype='float32')
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)  # Convert mono shape (samples,) to (1, samples)
        else:
            tensor = tensor.T  # Convert stereo shape (samples, channels) to (channels, samples)
        return tensor, sr

    # Monkey-patch torchaudio.load globally within this runtime context
    torchaudio.load = soundfile_load_fallback

    # Load the model with the compatible transformers version
    f5_model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, token=True).to("cuda")
    
    segment_files = []
    for i, chunk in enumerate(chunks):
        out_path = os.path.join(workspace, f"seg_{i}.wav")
        duration = chunk["end"] - chunk["start"]
        
        audio_array = f5_model(
            chunk["translated_text"],
            ref_audio_path=ref_voice,
            ref_text="." 
        )
        
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        sf.write(out_path, np.array(audio_array, dtype=np.float32), samplerate=24000)
        
        segment_files.append((out_path, chunk["start"], duration))
        
    del f5_model
    torch.cuda.empty_cache()
    return segment_files

def run_phase_5_mux(segment_files, background_audio, original_video, output_video, target_lang):
    print("[+] Phase 5: Rebuilding timeline with perfect lip-sync padding...")
    workspace = os.path.dirname(background_audio)
    
    concat_list_path = os.path.join(workspace, "concat.txt")
    current_time = 0.0
    import soundfile as sf
    
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for i, (file_path, start_time, _) in enumerate(segment_files):
            # 1. Calculate the exact silence gap needed before this clip plays
            gap = start_time - current_time
            if gap > 0:
                silence_path = os.path.join(workspace, f"silence_{i}.wav")
                # Generate a purely silent WAV file matching IndicF5's 24000Hz Mono format
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                    "-t", str(gap), silence_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                f.write(f"file '{os.path.abspath(silence_path).replace(chr(92), '/')}'\n")
            
            # 2. Write the actual localized voice clip to the timeline
            f.write(f"file '{os.path.abspath(file_path).replace(chr(92), '/')}'\n")
            
            # 3. Update the timeline tracker with the ACTUAL duration of the generated audio
            gen_data, gen_sr = sf.read(file_path)
            actual_gen_duration = len(gen_data) / gen_sr
            current_time = max(start_time, current_time) + actual_gen_duration
            
    # 4. Stitch the silence and dialogue chunks into one continuous master track
    dubbed_vocals = os.path.join(workspace, "dubbed_vocals.wav")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", concat_list_path, "-c", "copy", dubbed_vocals
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 5. Mix the new master vocal track with the original background score
    mixed_audio = os.path.join(workspace, "final_dubbed_track.ac3")
    subprocess.run([
        # Notice background_audio is passed FIRST so the duration matches the exact movie length
        "ffmpeg", "-y", "-i", background_audio, "-i", dubbed_vocals,
        "-filter_complex", "amix=inputs=2:duration=first:dropout_transition=0", "-c:a", "ac3", mixed_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 6. Mux back into the final video container
    lang_code = "tel" if target_lang.lower() == "telugu" else "hin"
    subprocess.run([
        "ffmpeg", "-y", "-i", original_video, "-i", mixed_audio,
        "-map", "0:v", "-map", "0:a", "-map", "1:a",
        "-c:v", "copy", "-c:a:0", "copy", "-c:a:1", "ac3",
        f"-metadata:s:a:1", f"language={lang_code}",
        f"-metadata:s:a:1", f"title={target_lang} (AI Dubbed)",
        output_video
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"\n[++] SUCCESS: Finished movie generated at: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--output", required=True)
    
    args = parser.parse_args()
    workspace_dir = "E:\\local-dubber\\workspace"
    
    setup_environment(workspace_dir)
    vocals, background = run_phase_1_demucs(args.input, workspace_dir)
    timeline_chunks = run_phase_2_transcribe(vocals)
    translated_chunks = run_phase_3_translate(timeline_chunks, args.lang)
    generated_segments = run_phase_4_synthesis(translated_chunks, args.voice, workspace_dir)
    
    run_phase_5_mux(generated_segments, background, args.input, args.output, args.lang)