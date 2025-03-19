# RedScribe

RedScribe is a fully offline audio transcription tool for red teamers, penetration testers, or anyone who wants to transcribe and summarize recorded meetings.  

RedScribe:  
✅ Transcribes MP3 meeting recordings with [Faster Whisper](https://github.com/SYSTRAN/faster-whisper).  
✅ Summarizes key points & action items using a local LLM (Ollama + Mistral-Orca).  
✅ Runs as a CLI tool via Pipx, keeping dependencies isolated.  

## Running RedScribe  

### Basic Usage  

```bash
$ redscribe -h                                                  
usage: redscribe [-h] [--transcript TRANSCRIPT] [--summary SUMMARY] [--model MODEL] [--prompt-file PROMPT_FILE] mp3_file

RedScribe: Transcribe and summarize red team scoping calls.

positional arguments:
  mp3_file              Path to the MP3 file

options:
  -h, --help            show this help message and exit
  --transcript TRANSCRIPT
                        Output transcript file name (optional)
  --summary SUMMARY     Output summary file name (optional)
  --model MODEL         Whisper model name (default: large-v2)(options:medium.en,large-v2,turbo,large)
  --prompt-file PROMPT_FILE
                        Path to a custom prompt file for summarization (optional)
```  

To process an MP3 audio recording of a meeting or engagement, run the following command. This will generate a transcript file `meeting_final-transcription.txt` as well as a summary with action items as `meeting_summary.txt`. 

```bash
redscribe meeting.mp3
```  

You can also include the optional `--transcript` and/or `--summary` flags to name your transcript and summary files. Otherwise, they will use the base name of the provided MP3 file.

```bash
redscribe meeting.mp3 --transcript ClientName_transcript.txt --summary ClientName_summary-w-action-items.txt
```  

If you would like to specify a different Whisper model than the default (`large-v2`), include the optional `--model` flag.  

```bash
redscribe meeting.mp3 --summary ClientName_summary.txt --model medium.en
```  

### Prompting Mistral-Orca  

RedScribe uses the following hard-coded prompt when summarizing the meeting transcription:  

```text
Please analyze the following meeting transcript and provide a structured summary.

### **Summary of the Meeting**  
- Summarize the main discussion points concisely.  
- Identify any critical insights shared by participants.

### **Action Items & Decisions**  
- List any tasks or follow-ups mentioned in the meeting.  
- Clearly specify who is responsible for each action item (if mentioned).  
- Highlight any unresolved issues or decisions that require further discussion.
```  

If you would like to provide RedScribe with a custom prompt for summarization, you can do so with the `--prompt-file` flag.  

Add your custom prompt to a file of your choosing:  

```text
# cust-prompt.txt
Summarize this transcript, focusing on any security vulnerabilities mentioned.

List any tasks or follow-ups mentioned in the meeting.
```  

Then run `redscribe`:  

```bash
redscribe meeting.mp3 --prompt-file /path/to/cust-prompt.txt
```  

## Installation  

### Considerations  

RedScribe works best when GPU support is enabled. If you're not able to utilize your GPU (or have an underpowered one), RedScribe will always fall back to CPU.  

You're responsible for configuring CUDA support for your specific instance. A brief set of generic instructions is provided below:  

OS: Pop_OS! 22.04
GPU: NVIDIA RTX 2000 Ada w/ 8GB VRAM & 3072 CUDA Cores  

Install CUDA GPU Inference and Deep Neural Network Library:  

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -O ~/Downloads/cuda-keyring_1.0-1_all.deb
sudo dpkg -i ~/Downloads/cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8 libcudnn8 libcudnn8-dev libcudnn9-cuda-12 libcudnn9-dev-cuda-12

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.zshrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```  

### Install Pipx  

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```  

### Install System Dependencies  

```bash
sudo apt update
sudo apt install -y git curl ffmpeg
```  

### Install and Configure Ollama  

RedScribe uses Ollama to run Mistral, providing fast and efficient LLM processing that is local to your machine.  

1. Install Ollama:  

```bash
curl -fsSL https://ollama.com/install.sh | sh
```  

-> Configure for GPU support:  

```bash
mkdir ~/.ollama
vim ~/.ollama/config
```  

-> Add the following to `~/.ollama/config`:  

```ini
[core]
use_gpu = true
```  

-> Verify installation:  

```bash
ollama --version
```  

2. Download and setup Mistral:  

```bash
ollama pull mistral-openorca:7b-q8_0
```  

Note, the `mistral-openorca:7b-q8_0` model is currently (March 2025) the best option as it offers a good balance between performance and resource efficiency at 7b parameters, has quantization (8-bit wights for higher accurace), has better context retention for long transcriptions, and is optimized for summarization and structured responses.

-> Test the model:  

```bash
ollama run mistral-openorca:7b-q8_0 "<PASTE_TEXT_TO_SUMMARIZE>"
```  

### Install RedScribe via Pipx  

Install RedScribe as a standalone CLI tool:  

```bash  
pipx install git+https://gitlab.com/al3631/redscribe.git
```

Or for development, install usign the followin steps:  

```bash
git clone https://gitlab.com/al3631/redscribe
cd redscribe
pipx install .
```  

***Note:*** Install may take a while due to PyTorch.  

## License  

RedScribe is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
