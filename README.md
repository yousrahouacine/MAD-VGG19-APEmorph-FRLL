# MAD-VGG19-APEmorph-FRLL

This repository contains the code used for Morph Attack Detection (MAD) experiments
based on a VGG19 model trained on the APEmorph dataset, with cross-dataset evaluation
performed on the FRLL dataset.

## Overview
The experimental pipeline includes:
- Identity-based train/test split on APEmorph
- Class balancing within each split
- Training of a VGG19-based classifier on APEmorph
- Embedding extraction using MTCNN
- Cross-dataset evaluation on FRLL
- t-SNE visualization of embeddings




