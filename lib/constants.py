# Sample rate that all audio files are loaded in
SAMPLE_RATE = 22050

# Length of audio, in samples
# Used in adjust_length in helpers - all audio files will be truncated/zero-padded to this length when they're loaded!
AUDIO_LENGTH = round(SAMPLE_RATE * 4)
