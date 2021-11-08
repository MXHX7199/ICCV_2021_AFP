<div style="text-align:center">
<img src="sth\AFP.png" alt="AFP_Logo" width="700"/>
<h2>Improving Neural Network Efficiency via Post-training Quantization with Adaptive Floating-Point</h2>
</div>
pytorch implementation of Adaptive Floating-Point for model quantization

# Overview
## Description
- afp_sgd  (`/code/afp_sgd.py`)

A modified SGD model that specifies the quantized weight bits for each weight matrix. The only difference is that a `weight_bits` parameter should be provided. The `params` parameter also accept dicts with `weight_bits` keys.
```python
optimizer = AFP_SGD(params=model.parameters(), 
                    lr=0.1, 
                    momentum=0.9, 
                    weight_bits=3)
inq_scheduler = AFPScheduler(optimizer)
inq_scheduler.step()
optimizer.zero_grad()
loss_fn(model(input), target).backward()
optimizer.step()
```

- quantize_scheduler (`/code/quantize_scheduler.py`)

Contains the functions that handles the quantization of weights using Adaptive Floating-Point format.

 - quantize_weight(weight, possible_quantized)
 
Quantize a single `weight` into the nearest neighbour in `possible_quantized`.
 - AFPScheduler(object)
 
A class that decides the quantize range of all weight matrices in an optimizer and provide quantize API.

Initialization:  
`__init__(self, optimizer: AFP_SGD)`  
Accepts an `AFP_SGD` optimizer that specifies the weight bits for each weight matrix and decides the possible quantized values of each weight matrix adaptively according to the range of the weight matrix and weight bits.

`step(self)`  
An quantization API that execute quantization procedure.

Usage:
```python
optimizer = AFP_SGD(...)
inq_scheduler = AFPScheduler(optimizer)
validate(...)   # pre-quantization validation
inq_scheduler.step()
validate(...)   # post-quantization validation
```

- getSA (`/code/getSA.py`)
 - compute_KL(p, E_e, E_s)


Compute the KL-divergence of weight matrix p and quantized one given exponent bit-width E_e and mantissa bit-width E_s

 - getQuanMSE(N_q, E_e, resume=None)


Compute average KL-divergence of a model loaded from `resume` given total quantization bit-width `N_q` and exponent bit-width `E_e`.

 - SA(object)


The simulation annealing class that finds the optimal bit-width of the exponent to minimize the average KL-divergence, given the model and target quantization bit-width. The searching algorithm can be substituted with other ones such as genetic searching,  bayesian optimization (used in our paper).


## Citation

We now have a [paper](#), titled "Improving Neural Network Efficiency via Post-training Quantization with Adaptive Floating-Point", which is published in ICCV-2021.
```bibtex
@inproceedings{liu2021afp,
 title={Improving Neural Network Efficiency via Post-training Quantization with Adaptive Floating-Point},
 author={Liu, Fangxin and Zhao, Wenbo and He, Zhezhi and Wang, Yanzhi and Wang, Zongwu Wang and Dai, Changzhi and Liang, Xiaoyao and Jiang, Li},
 booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
 year={2021}
}
```

## To-do

- [ ] ***Coming soon:*** Updated Code.
