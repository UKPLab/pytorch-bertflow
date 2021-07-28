# Pytorch-bertflow

This is an re-implemented version of BERT-flow using Pytorch framework, which can reproduce the results from [the original repo](https://github.com/bohanli/BERT-flow).

## Usage
Please refer to the simple example [./example.py](./example.py)
```python
python example.py
```
## Note
- Please shuffle your training data, which makes a huge difference.
- The pooling function makes a huge difference in some datasets (especially for the ones used in the paper). To reproduce the results, please use 'first-last-avg'.