# VAMP-Diff: VampPrior Latent Diffusion for Photoplethysmography Modeling
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)
> A jointly trained variational diffusion model for 
> physiologically realistic PPG signal generation.
---
## 📄 Paper
**VAMP-Diff: VampPrior Latent Diffusion for Photoplethysmography Modeling**  
Fatemeh Ghasemi Balouei, Nathan Willemsen, Mahesh Banavar, Bahman Moraffah  
*Asilomar Conference on Signals, Systems, and Computers 2026*
📥 [Download Full Paper (PDF)](./VAMP_Diff__VampPrior_Latent_Diffusion_for_Photoplethysmography_Modeling.pdf)
---
## 🧠 Overview
PPG generation is harder than it looks. A generative model for PPG 
must simultaneously reproduce cardiac periodicity at 0.7–3 Hz, 
respiratory modulation at 0.1–0.5 Hz, and realistic beat morphology 
including the systolic upstroke and dicrotic notch — all in a single 
10-second window. Standard VAE approaches fail at this because:

1. **A global pooled latent loses temporal structure.** Encoding a 
   full PPG window into a single vector collapses the beat-to-beat 
   rhythm and respiratory envelope into one fixed-size summary. We 
   showed this increases HR absolute error from 0.56 bpm (ours) to 
   3.5 bpm (vanilla VAE).

2. **A Gaussian prior causes generation failure.** The standard VAE 
   prior N(0,I) does not match the learned posterior over PPG signals. 
   Sampling from N(0,I) at generation time produces out-of-distribution 
   latents that the decoder maps to noise or physiologically invalid 
   signals.

3. **A diffusion decoder alone is not enough.** Without a structured 
   prior, a VAE-Diffusion hybrid still fails at unconditional generation 
   because the prior-posterior mismatch problem persists regardless of 
   the decoder architecture.

VAMP-Diff addresses all three problems with three corresponding design 
choices:

**Temporal sequence latent.** Instead of pooling z to a vector, the 
encoder maps a (1, 3072) PPG window to a full sequence latent 
z ∈ R^{256×768}, preserving temporal periodicity information at every 
position. This is what enables HR-faithful generation — the decoder 
has access to the full temporal structure of the encoded signal.

**VampPrior with full-resolution construction.** The VampPrior 
replaces N(0,I) with a mixture of K=100 learned pseudo-input posteriors, 
initialized from real training windows using stratified sampling across 
HR and amplitude bins. Critically, we construct the prior in the full 
(256×768) latent space rather than in a compressed representation — 
this was the key fix that reduced the HR gap from 13.2 bpm (compressed 
prior) to 1.2 bpm (full-resolution prior). The compressed prior loses 
temporal periodicity information during downsampling; the full-resolution 
prior preserves it.

**FiLM-conditioned diffusion decoder.** The 1D U-Net decoder receives 
the sequence latent z through two complementary pathways: direct spatial 
injection at each U-Net resolution level, and global FiLM modulation 
combined with the diffusion timestep embedding at every residual block. 
This allows z to guide both the overall waveform shape and fine 
morphological details simultaneously.

**Auxiliary losses for morphological fidelity.** A diffusion loss alone 
produces signals that are statistically reasonable but morphologically 
blurry. We add spectral, derivative, amplitude, and peak-to-peak losses 
to supervise specific morphological properties explicitly, reducing peak 
timing error and improving HR estimation from reconstructed signals.

### Key results on CapnoBase test set

| Metric | Value |
|--------|-------|
| Reconstruction Corr | 0.9991 |
| HR Abs Error (recon) | 0.56 bpm |
| HR Gap (generation, 5000 samples) | 1.2 bpm |
| Mean Pairwise Distance (diversity) | 3.99 ± 0.65 |
| Anomaly detection AUROC (noise) | 1.000 |
| Anomaly detection AUROC (overall) | 0.739 |
---
## 📊 Dataset
[CapnoBase](https://doi.org/10.5683/SP2/MR0PTF) — 42 ICU patients,
PPG and EtCO₂ capnography at 300 Hz.
---
## 🗂️ Code
The full implementation is in [`vamp-diff/`](./vamp-diff/). See the [code README](./vamp-diff/README.md) for setup and usage instructions.
---
## 📝 Citation
```bibtex
@inproceedings{ghasemibalouei2026vampdiff,
  title     = {VAMP-Diff: VampPrior Latent Diffusion for Photoplethysmography Modeling},
  author    = {Ghasemi Balouei, Fatemeh and Willemsen, Nathan and Banavar, Mahesh and Moraffah, Bahman},
  booktitle = {Asilomar Conference on Signals, Systems, and Computers},
  year      = {2026}
}
```
