# Changelog  

## 1.0.1 - 2025-03-19  

### Added  

- Added ability to specify a prompt file over the default, hard-coded prompt.
- Clear CUDA memory after each run to help prevent OOM issues.

### Changed  

- Switched OpenAI-Whisper with Faster-Whisper to avoid file size limiitations and constant issues with GPU OOM when using large models for transcription.
- Improved transcription for a more conversational look/feel and removed timestamps.
- Updated default model to 'large-v2' from 'medium.en' as it performs better than medium.en and is much more accurate.

### Removed  

- Removed nused diarization functions and dependencies.
- Removed unused redscribe/logger.py. Might look into logging in the future but it's not really needed right now.

## 1.0.0 - 2025-03-18  

_Initial Release.