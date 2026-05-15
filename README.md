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

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- scipy, numpy, matplotlib

---

## 🗂️ Repository Structure
