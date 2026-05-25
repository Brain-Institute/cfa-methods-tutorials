# Ophthalmo Corp — Methods Tutorials

These tutorials have been created by **Ophthalmo Corp**. Each tutorial is fully self-contained and modular, designed to be used independently for various Medical Preprocessing and AI tasks.

---

## Tutorials

### 1. Stereo to Mono
**Directory:** `Stero to mono/`

This tutorial teaches how to convert **stereo** audio (2 channels) into **mono** audio (1 channel) and resample it to **16 kHz** — a common preprocessing step before speech recognition. Many ASR models (including those used in medical transcription) require single-channel audio at a fixed sample rate. The tutorial implements a complete preprocessing pipeline: load a WAV file, convert to a PyTorch tensor, average the left and right channels into mono, resample to 16 kHz, and return a 1D NumPy array ready for downstream use. Waveform visualizations are included to illustrate the before-and-after effect of channel averaging.

**What you will learn:** WAV file structure, mono vs. stereo formats, key audio terminology (sample rate, amplitude, resampling, downmixing), and how to use `soundfile`, `torch`, and `torchaudio` to build the pipeline.

- Notebook: `Methods_Tutorial_Stereo_to_Mono.ipynb`
- Includes demo stereo and mono `.wav` files for hands-on testing
- Dependencies: `requirements.txt` (`numpy`, `soundfile`, `torch`, `torchaudio`, `matplotlib`)

---

### 2. Word Error Rate (WER)
**Directory:** `Word Error Rate (WER)/`

This tutorial teaches how to calculate **Word Error Rate (WER)** — the standard metric for evaluating the accuracy of speech recognition (ASR) systems. WER quantifies how many words a system got wrong by counting the minimum number of word-level edits (substitutions, insertions, and deletions) needed to turn the system's output into the correct transcript, divided by the total number of words in the reference. The tutorial covers text normalization (lowercasing, punctuation removal), single-pair and batch WER computation, edge case handling, and visualization of WER distributions across a dataset of medical transcriptions.

**What you will learn:** The WER formula (S + I + D) / N, the three edit operation types, why text normalization is standard practice before comparison, how to use the `jiwer` library, and how to interpret and visualize results.

- Notebook: `Methods_Tutorial_WER.ipynb`
- Dependencies: `requirements.txt` (`jiwer`, `matplotlib`)
