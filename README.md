<div align="center">

# :robot: Yet Another Transformer Implementation :robot:

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)


</div>

>[!WARNING]
> This repository was developed for academic and personal purposes in order to better understand the underlying architecture of the Transformer and to use it for future small projects.

---

## :package: Installation
First, clone the repo.
```bash
git clone https://github.com/RistoAle97/yati
```
Then, install the dependencies.
```bash
pip install -r requirements.txt
```
---

## :hammer_and_wrench: Implementation details
>[!NOTE]
> Some implementation choices have been made that may differ from the original paper:
> - The source and target embeddings are shared, so a unified vocabulary (one for all the languages in a NMT task to give an example) is needed.
> - The embeddings are tied to the linear output (i.e.: they share the weights).
> - Pre-normalization was employed instead of post-normalization.
> - Layer normalization is performed at the end of both the encoder and decoder stacks.
> - There is no softmax layer as it is already used by the CrossEntropy loss function implemented in PyTorch.

Hereafter a comparison between the original transformer and the one from this repository.
Original             | This repository
:-------------------------:|:-------------------------:
<img src="https://github.com/RistoAle97/yati/blob/main/assets/transformer_original.jpg" width=80%> | <img src="https://github.com/RistoAle97/yati/blob/main/assets/model_architecture.jpg" width=78%/>

---

## :books: Bibliography
Work in progress.

## :memo: License
This project is [MIT licensed](https://github.com/RistoAle97/yati/blob/main/LICENSE).
