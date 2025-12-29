# S2-FSAM
Single-Step Friendly Sharpness Aware Minimization

Implementation combines \
S2-SAM https://openreview.net/pdf?id=MJgMMqMDu4 (NeurIPS 2024) and \
F-SAM https://arxiv.org/pdf/2403.12350

To my knowledge, no one ever combined FSAM with S2SAM, so there you go, bleeding-edge tech!

<img src="science.png" alt="Yeah, science" width="300">


I use a custom base optimizer so you might need to change the `step()` function to add `closure` and whatnot.

