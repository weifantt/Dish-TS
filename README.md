# Dish-TS

Source code for the paper,
["Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting"](https://arxiv.org/abs/2302.14829),
in AAAI 2023.

## Overview
Dish-TS is a general paradigm for time series forecasting against distribution shift.

## Usage
Similar to [reversible instance normalization](https://openreview.net/forum?id=cGDAkQo1C0p), Dish-TS is model-agnostic such that it can be coupled with any forecasting architectures.

Note that in experiments, we directly take the original data for training/evaluation to directly reflect the distribution shift in time series, and do not use preprocessing techniques (e.g., z-score normalization, min-max normalization) to process time series dataset.

## Citation
If you find our work interesting, you can the paper as

```text
@inproceedings{fan2023dish,
  title={Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting},
  author={Fan, Wei and Wang, Pengyang and Wang, Dongkun and Wang, Dongjie and Zhou, Yuanchun and Fu, Yanjie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={6},
  pages={7522--7529},
  year={2023}
}
```
