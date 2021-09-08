# Pytorch-bertflow

This is an re-implemented version of BERT-flow using Pytorch framework, which can reproduce the results from [the original repo](https://github.com/bohanli/BERT-flow). This code is used to reproduce the results in [the TSDAE paper](https://arxiv.org/abs/2104.06979).

## Usage
Please refer to the simple example [./example.py](./example.py)
```python
python example.py
```
## Note
- Please shuffle your training data, which makes a huge difference.
- The pooling function makes a huge difference in some datasets (especially for the ones used in the paper). To reproduce the results, please use 'first-last-avg'.

## Contact
Contact person and main contributor: [Kexin Wang](https://kwang2049.github.io/), kexin.wang.2049@gmail.com

[https://www.ukp.tu-darmstadt.de/](https://www.ukp.tu-darmstadt.de/)

[https://www.tu-darmstadt.de/](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
