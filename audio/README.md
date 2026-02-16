# Audio Experiment

## Appendix I: Cross-Modal Generalization to Audio Signals

The HPU-Core architecture described in previous sections was trained exclusively on synthetic Hamiltonian dynamics on a 2D torus. It never encountered audio data during training. Yet when I feed it spectrograms derived from real audio signals, it reconstructs them with surprising fidelity.

I used the same checkpoint from Seed 32 (epoch 7196) that produced the visual results in Appendix H. The only modification was preprocessing: I convert audio waveforms to magnitude spectrograms via STFT, treat these 2D time-frequency representations as field configurations, pass them through the spectral layers, and invert back to audio.

The results are shown in the accompanying figures. The spectrogram reconstruction preserves harmonic structure and temporal envelope. The waveform comparison reveals that the reconstructed signal tracks the original with mean absolute error below 0.05 in normalized amplitude. Listening tests confirm the reconstruction is perceptually nearly identical to the source, despite the model having no prior exposure to acoustic data.

This is not supposed to happen. The model learned Hamiltonian evolution on an abstract manifold, not audio compression. Yet the same operator that segments my desktop screenshot also reconstructs speech and music. I interpret this as evidence that the Hamiltonian crystal has learned a general-purpose field dynamics prior—one that happens to be useful for any signal representable as a 2D field, whether luminance or log-magnitude spectrograms.

The model size is 2.7 MB. It runs in real time on CPU. It was not trained for this task.

What I do not know: whether this generalization holds for all audio types, whether performance degrades on out-of-distribution acoustic environments, or whether the reconstruction quality reflects genuine information preservation or merely plausible generation constrained by the Hamiltonian dynamics. I also have not measured perceptual metrics like PESQ or STOI; the "nearly identical" claim is based on informal listening and waveform inspection.

The code for this experiment is in the repository under `audio` directory. It reuses the same `HamiltonianNeuralNetwork` class without architectural changes.

If an abstract physical system can reconstruct audio and segment images without specific training, then the boundary between modalities is artificial. What matters is the underlying structure of the operator. This result suggests that certain types of "compression" or "information processing" are universal and emerge from dynamics, not from the data.
