# taxophoney

GPT (Decoder only Transformer - from scratch) generated fake/phoney taxonomies, trained on NCBI taxonomy dataset, included in this repository.

## Requirements

- Pytorch (with CUDA support - for reasonably short training runs)


## Quick training results

```bash
$ python gpt.py
Using device : cuda
step 0: train loss 4.4625, val loss 4.4653
step 500: train loss 2.0843, val loss 2.1280
step 1000: train loss 1.5394, val loss 1.5920
step 1500: train loss 1.3097, val loss 1.3789
step 2000: train loss 1.1842, val loss 1.2741
step 2500: train loss 1.1017, val loss 1.2182
step 3000: train loss 1.0408, val loss 1.1938
step 3500: train loss 0.9831, val loss 1.1692
step 4000: train loss 0.9382, val loss 1.1591
step 4500: train loss 0.8935, val loss 1.1392
step 4999: train loss 0.8545, val loss 1.1383
```
