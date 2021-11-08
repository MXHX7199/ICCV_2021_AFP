import math
from functools import partial
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from bisect import bisect_right


def quantize_weight(weight, possible_quantized):
    """Quantize a single weight using AUSN quantization scheme.
    """

    abs_weight = math.fabs(weight)
    pos = bisect_right(possible_quantized, abs_weight)

    if pos == 0:
        quantized = 0 if abs_weight < possible_quantized[0] / 2 else possible_quantized[0]
    elif pos == len(possible_quantized):
        quantized = possible_quantized[-1]
    else:
        left = possible_quantized[pos - 1]
        right = possible_quantized[pos]
        quantized = left + round((abs_weight - left) / (right - left)) * (right - left)

    quantized = math.copysign(quantized, weight)
    return quantized


class AFPScheduler(object):
    """Handles the the weight quantization scheme of AUSN

    Args:
        optimizer (Optimizer): Wrapped optimizer (use inq.SGD).

    Example:
        >>> optimizer = inq.SGD(...)
        >>> inq_scheduler = AFPScheduler(optimizer)
        >>> for inq_step in range(3):
        >>>     inq_scheduler.step()
        >>>     for epoch in range(5):
        >>>         train(...)
        >>> inq_scheduler.step()
        >>> validate(...)

    """

    def __init__(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        for group in self.optimizer.param_groups:
            group['ns'] = list()  # stores the upper and lower bound of each layer in `group', list of tuples.
            group['possible_quantized_s'] = list()
            if group['weight_bits'] is None:
                continue
            for p in group['params']:
                if p.requires_grad is False:
                    group['ns'].append((0, 0))
                    group['possible_quantized_s'].append(list())
                    continue

                lower, upper = 0, 0
                possible_quantized = list()

                # print(group['weight_bits'])
                if group['weight_bits'] == 3:
                    s_bit = 0           # s_bit, bit for superposition
                    ratio = 1
                elif group['weight_bits'] == 4:
                    s_bit = 1
                    ratio = [1, 1.5]
                else:
                    s_bit = 2
                    ratio = [1, 1.25, 1.5, 1.75]

                main_bit = group['weight_bits'] - 1 - s_bit     # 1 sign bit
                abs_max = torch.max(torch.abs(p.data)).item()
                upper = math.ceil(math.log2(abs_max / max(ratio)))   # available main part: [lower, upper]
                # max(ratio) * 2 ** upper >= max
                lower = upper + 1 - 2 ** main_bit

                possible_main = list(map(lambda x: 2 ** x, range(lower, upper + 1)))
                for i in possible_main:
                    for r in ratio:
                        possible_quantized.append(i * r)

                possible_quantized = possible_quantized[1:]
                possible_quantized.append(0)

                group['ns'].append((lower, upper))
                group['possible_quantized_s'].append(possible_quantized)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        """Quantize the parameters handled by the optimizer.
        """
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue
                ns = group['ns'][idx]
                device = p.data.device
                possible_quantized = torch.tensor(group['possible_quantized_s'][idx]).to(device)

                abs_p = torch.abs(p).view(-1)
                shape = p.shape
                idxs = (abs_p.unsqueeze(0) - possible_quantized.unsqueeze(1)).abs().min(dim=0)[1]
                quantized_p = possible_quantized.data[idxs].view(shape)
                p.data = quantized_p * torch.sign(p)
        print('all parameters quantized')
