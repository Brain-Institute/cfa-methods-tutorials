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

---

### 3. Audio Transcription with Whisper
**Directory:** `Transcription_Whisper/`

This tutorial teaches how OpenAI's **Whisper large-v3-turbo** model takes an audio waveform as input and produces a text transcription as output. It walks through the complete inference pipeline using the real model — from loading the weights and preprocessing audio into mel-spectrogram features, to generating tokens and decoding the final text. The tutorial also covers the high-level `pipeline` API, language and task control (transcription vs. translation), sentence-level timestamps, and chunked long-form transcription for audio longer than Whisper's 30-second window. The model is ~809 M parameters (~1.6 GB download) and runs on both CPU and GPU.

**What you will learn:** The three components of a speech model (model, processor, tokenizer), how Whisper converts audio into a mel spectrogram, how greedy decoding generates transcription tokens deterministically, and how to use the `transformers` pipeline API for practical transcription tasks.

- Notebook: `Methods_Tutorial_Transcription_Whisper.ipynb`
- Requires: `Sample_Audio.mp3` (placed in the same directory as the notebook)
- Dependencies: `requirements.txt` (`torch`, `transformers`, `soundfile`, `scipy`, `numpy`, `matplotlib`, `accelerate`)

---

### 4. Fine-Tuning Whisper with Soft Prompts
**Directory:** `Fine-Tuning_Soft_Prompt_Whisper/`

This tutorial teaches how to adapt **OpenAI Whisper large-v3-turbo** using **soft prompt tuning** — a parameter-efficient method that freezes the entire Whisper model and instead learns a small set of trainable continuous vectors (the soft prompt) that are prepended to the decoder's input embeddings to steer generation. Rather than updating ~809 M model weights, only a tiny prompt matrix `P ∈ ℝ^(m × d)` is trained, where `m` is the prompt length and `d` is the decoder embedding dimension. The tutorial uses synthetic mock audio and transcripts so no external dataset is required, and verifies the training mechanics through a one-step smoke test that confirms gradients flow into the prompt and not into the frozen model.

**What you will learn:** What soft prompts are and how they differ from text prompts, how to freeze a model and attach trainable prompt embeddings, how to prepend prompt embeddings to decoder inputs, how to mask prompt positions in the loss with `-100`, and how soft prompt tuning compares to LoRA and full fine-tuning in terms of parameter count and expressiveness.

- Notebook: `Methods_Tutorial_Prompt_Tuning_Whisper.ipynb`
- Uses synthetically generated mock data — no external audio files required
- Dependencies: `requirements.txt` (`torch`, `transformers`, `numpy`, `matplotlib`)

---

### 5. Fine-Tuning Whisper with LoRA
**Directory:** `Fine-Tuning_LoRA_Whisper/`

This tutorial teaches how to fine-tune **OpenAI Whisper large-v3-turbo** using **LoRA (Low-Rank Adaptation)** — a parameter-efficient method that freezes the base model and inserts small trainable low-rank adapter matrices into selected attention projection layers. For a frozen weight matrix `W`, LoRA learns an update `ΔW = BA` where `A` and `B` are low-rank matrices controlled by rank `r`, making it more expressive than soft prompts while remaining far cheaper than full fine-tuning. The tutorial uses synthetic mock audio and transcripts, walks through attaching LoRA adapters to Whisper's attention layers, and runs a smoke test to confirm that only the adapter weights receive gradient updates.

**What you will learn:** How LoRA works mathematically, how to attach LoRA adapters to Whisper attention projections, how to prepare audio inputs and decoder labels for fine-tuning, how to inspect trainable vs. frozen parameters, and how LoRA compares to soft prompt tuning and full fine-tuning.

- Notebook: `Methods_Tutorial_Fine_Tuning_Whisper_LoRA.ipynb`
- Uses synthetically generated mock data — no external audio files required
- Dependencies: `requirements.txt` (`torch`, `transformers`, `numpy`, `matplotlib`)
