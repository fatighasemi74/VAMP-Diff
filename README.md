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
VAMP-Diff combines:
- A **temporal PPG encoder** that maps signals to a sequence latent
- A **conditional 1D diffusion decoder** conditioned on the full temporal latent
- A **VampPrior** defined on a compact pooled latent for data-adaptive generation
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
