# Author: Aaron Lesmeister
# Date: 2025-03-18
# Description: CLI for RedScribe, a tool to transcribe and summarize MP3 audio files from scoping calls/meeting recordings.
# License: Apache-2.0

# TODO
# - Research offline speaker diarization.
# - Research adding a '--forensic' flag to add timestamps to the transcript and transcription is created in a consistent manner. 
#     Forensic mode should increase beam_size between 5-10 and set the `word_timestamps` parameter to True. Note, higher beam_size
#     will exponentially increase the time it takes to transcribe audio and will also increase memory useage. This would need a
#     check to ensure the system has enough VRAM to handle the increased memory usage.
# - Add error handling
# - Add logging

import os
import re
import json
import argparse
import subprocess
from faster_whisper import WhisperModel
import torch
from rich import print
from rich.text import Text
from rich.markup import escape
from time import sleep
import logging
logging.getLogger("speechbrain.utils.quirks").setLevel(logging.WARNING)


def convert_mp3_to_wav(mp3_file, wav_file):
    # Converts MP3 to WAV using FFmpeg. WAV is apparently better for Whisper.

    print(f"[bold][red][🔍][/red] Converting [cyan]{escape(mp3_file)}[/cyan] to WAV...[/bold]")
    
    # Run FFmpeg with all output suppressed
    subprocess.run(
        ["ffmpeg", "-i", mp3_file, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_file],
        stdout=subprocess.DEVNULL,  # Suppress standard output
        stderr=subprocess.DEVNULL,  # Suppress errors
        check=True
    )

    sleep(1)

def transcribe_audio(wav_file, model_name, transcript_file):
    # Transcribes audio using OpenAI Whisper

    print(f"[bold][red][📝][/red] Transcribing [cyan]{escape(wav_file)}[/cyan] using Whisper model '{escape(model_name)}'...[/bold]")

    # Handle Large Model: Smart GPU & CPU Switching**
    if model_name.lower() in ["large", "large-v1", "large-v2", "large-v3", "turbo"]:
        if torch.cuda.is_available():
            try:
                # Attempt GPU first
                model = WhisperModel(model_name, device="cuda", compute_type="float16")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("[bold red][🔥] GPU out of memory! Switching to CPU...[/]")
                    torch.cuda.empty_cache() # Clear GPU memory
                    sleep(3) # Give it time to clear
                    model = WhisperModel(model_name, device="cpu", compute_type="int8")  # Retry on CPU
                else:
                    raise
        else:
            print("[bold red][❌] No GPU available. Using CPU for large model...[/]")
            model = WhisperModel(model_name, device="cpu")  # Use CPU if no GPU
    else:
        # For Small/Medium Model: Prefer GPU but fall back to CPU if necessary
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = WhisperModel(model_name, device=device)
        except RuntimeError:
            model = WhisperModel(model_name, device="cpu")  # Retry on CPU
   
    # Transcribe with Faster_Whisper. 
    segments, _ = model.transcribe(wav_file, language="en")

    # Format transcription for a natural conversation flow
    formatted_transcript = []
    buffer_sentence = ""
    last_text = ""

    for segment in segments:
        text = segment.text.strip()

        # Remove exact duplicate lines
        if text == last_text:
            continue  # Skip exact duplicates
        last_text = text  # Track last processed text

        # Merge short fragments into complete sentences
        if buffer_sentence:
            buffer_sentence += " " + text
        else:
            buffer_sentence = text

        # Ensure sentences are structured properly
        if text.endswith((".", "?", "!")):
            # Try to remove excessive repetitions
            buffer_sentence = re.sub(r'\b(\w+)\s+\1\b', r'\1', buffer_sentence)

            formatted_transcript.append(buffer_sentence.strip())  # Add full sentence to transcript
            buffer_sentence = ""  # Reset buffer

    if buffer_sentence:
        formatted_transcript.append(buffer_sentence)
    
    # Save transcript to a file
    print(f"[bold][red][💾][/red] Saving transcription to [cyan]{escape(transcript_file)}[/cyan]...[/bold]")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_transcript))

    sleep(1)
    return formatted_transcript

def summarize_transcript(transcript_file, summary_file, prompt_file=None):
    # Summarizes the transcript using Ollama.

    print(f"[bold][red][📝][/red] Summarizing transcript from [cyan]{escape(transcript_file)}[/cyan]...[/bold]")

    with open(transcript_file, "r") as f:
        transcript = f.read()

    # Check if user provided a custom prompt file
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                custom_prompt = f.read().strip()
            prompt = f"{custom_prompt}\n\n### Transcript:\n{transcript}"

            print(f"[bold][red][📄][/red] Using custom prompt from [cyan]{escape(prompt_file)}[/cyan]...[/bold]")

        except FileNotFoundError:
            print(f"[bold red][❌] Error: Prompt file '{escape(prompt_file)}' not found. Falling back to default prompt.[/]")
            prompt_file = None  # Reset prompt file to ensure default prompt is used

    # Default prompt if no prompt file is provided
    if not prompt_file:
        prompt = f"""
Please analyze the following meeting transcript and provide a structured summary.

### **Summary of the Meeting**  
- Summarize the main discussion points concisely.  
- Identify any critical insights shared by participants.

### **Action Items & Decisions**  
- List any tasks or follow-ups mentioned in the meeting.  
- Clearly specify who is responsible for each action item (if mentioned).  
- Highlight any unresolved issues or decisions that require further discussion.

### **Transcript:**  
{transcript}
"""
    # Summarize the transcript using Ollama
    summary = subprocess.run(
        f'echo {json.dumps(prompt)} | ollama run mistral-openorca:7b-q8_0',
        shell=True, capture_output=True, text=True
        ).stdout
    
    # Save the summary to a file
    print(f"[bold][red][💾][/red] Saving summary to [cyan]{escape(summary_file)}[/cyan]...[/bold]")
    with open(summary_file, "w") as f:
        f.write(summary)
    sleep(1)

def cleanup_temp_files(wav_file):
    # Deletes temporary WAV file after processing.
    if os.path.exists(wav_file):
        os.remove(wav_file)

def cli_banner():
    # Prints the RedScribe ASCII banner.
    print("""
[bold red]\n
██████╗ ███████╗██████╗ ███████╗ ██████╗██████╗ ██╗██████╗ ███████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗██║██╔══██╗██╔════╝
██████╔╝█████╗  ██║  ██║███████╗██║     ██████╔╝██║██████╔╝█████╗  
██╔══██╗██╔══╝  ██║  ██║╚════██║██║     ██╔══██╗██║██╔══██╗██╔══╝  
██║  ██║███████╗██████╔╝███████║╚██████╗██║  ██║██║██████╔╝███████╗
╚═╝  ╚═╝╚══════╝╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═════╝ ╚══════╝
    (For Red Team Engagement Audio Summaries) github.com/kopfjager007
[/bold red]
""")

def main():
    # Main function for RedScribe.

    parser = argparse.ArgumentParser(description="RedScribe: Transcribe and summarize red team scoping calls.")
    parser.add_argument("mp3_file", help="Path to the MP3 file")
    parser.add_argument("--transcript", help="Output transcript file name (optional)")
    parser.add_argument("--summary", help="Output summary file name (optional)")
    parser.add_argument("--model", help="Whisper model name (default: large-v2)(options:medium.en,large-v2,turbo,large)", default="large-v2")
    parser.add_argument("--prompt-file", help="Path to a custom prompt file for summarization (optional)")

    args = parser.parse_args()
    
    # Determine output file names
    base_name = os.path.splitext(os.path.basename(args.mp3_file))[0]
    transcript_file = args.transcript or f"{base_name}_final-transcription.txt"
    summary_file = args.summary or f"{base_name}_summary.txt"
    wav_file = f"{base_name}.wav"

    cli_banner()

    # Convert MP3 to WAV
    convert_mp3_to_wav(args.mp3_file, wav_file)

    # Transcribe Audio
    transcription = transcribe_audio(wav_file, args.model, transcript_file)

    # Summarization
    summarize_transcript(transcript_file, summary_file, args.prompt_file)

    # Cleanup Temporary Files
    cleanup_temp_files(wav_file)

    print("[bold][red][✅][/red] RedScribe processing [green]complete[/green]![/bold]")

    # Clear CUDA memory after each run to help prevent OOM errors with subsequent runs.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
