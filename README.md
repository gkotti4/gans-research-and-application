# **GANs_Research-Project**
Research project exploring **NumPy**, **DCGAN**, and **optimized GAN** implementations for **2D procedural content generation**.

> **Note:** This is a reupload of academic and independent research conducted in 2024.
> The project is not under active development but remains available for reference and portfolio purposes.
> The repository structure is organized to separate experimental implementations, reports, and supporting material.

---

## Overview
This project investigates the evolution of Generative Adversarial Networks (GANs) — from NumPy-based prototypes to deep convolutional architectures (DCGAN) — to understand their behavior, stability, and generative capabilities in the context of procedural content generation (PCG) for games and visual synthesis.

The experiments focus on:
- Implementing a **GAN from scratch** using only NumPy (no frameworks).  
- Developing and training **DCGAN** and **optimized GAN** models in PyTorch.  
- Analyzing training stability, loss behavior, and image quality across architectures.  
- Exploring potential applications for **2D tile generation** and **texture synthesis**.

---

## **Project Structure**
GANs_Research_Project/  
│  
├── Source Code (local)/     # Local NumPy-based GAN implementation  
├── Presentation/            # Slide deck used for academic presentation  
├── Research Paper/          # IEEE-formatted research paper (unpublished)  
├── Literature Survey/       # Literature review and related work summary  
├── Diagrams/                # Training architecture and loss visualization  
└── External Links/          # Colab DCGAN + optimized GAN implementations (COLAB)  
  
---

## **Implemented Architectures**
| Model Type | Framework | Description |
|-------------|------------|-------------|
| **Vanilla GAN** | NumPy | Handcrafted forward and backward propagation, manual weight updates |
| **DCGAN** | PyTorch (Colab) | Convolutional generator and discriminator with batch normalization |
| **Optimized GAN** | PyTorch (Colab) | Improved stability using tuning and regularization |

Each model follows the same **generator–discriminator paradigm**, with varying complexity and training efficiency.

---

## **Results & Observations**

- The NumPy GAN successfully demonstrates the core adversarial training loop and convergence behavior.  
- DCGAN produces coherent images after sufficient epochs, revealing learned latent patterns.  
- Optimized models exhibit improved loss stability and reduced mode collapse.  

Visual examples and generated outputs are provided in the **presentation** and **diagrams** folders.

---

## **Reports & Presentation**

- `research_paper/`: IEEE-formatted paper describing methodology and results.  
- `literature_survey/`: Research summary and contextual review.  
- `presentation/`: Academic presentation slides summarizing project findings.

---

## **Author**

**George Kotti**  
B.S. in Computer Science & Engineering (AI + Systems)  
Focus Areas: Machine Learning / classical AI, Systems Engineering

---

## **References**

> *Selected key references used in this project.*

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Online Book](https://www.deeplearningbook.org/)
- Koprinkova-Hristova, P. (2020). *Machine Learning for Computer Scientists and Data Analysts*. Springer.
- Goodfellow et al., *Generative Adversarial Nets* (NeurIPS 2014)  
- Radford et al., *Unsupervised Representation Learning with DCGANs* (ICLR 2016)  
- Arjovsky et al., *Wasserstein GAN* (2017)  

---

### *Summary*

This repository encapsulates the learning progression from foundational GAN implementation in NumPy to modern DCGAN architectures — bridging theoretical understanding with practical experimentation for generative content design.
