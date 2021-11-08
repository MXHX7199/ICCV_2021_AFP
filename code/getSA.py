import numpy as np
import random
import torch
import os, sys
import math


if len(sys.argv) > 1:
    file = open(sys.argv[1], 'w+')
else:
    file = open("output.txt", 'w+')


def compute_KL(p, E_e, E_s):
    """ Compute the KL-divergence of weight matrix p and quantized one
        given exponent bit-width E_e and mantissa bit-width E_s

        :param p: weight matrix
        :param E_e: exponent bit-width
        :param E_s: mantissa bit-width
    """
    ratio = np.linspace(1., 2., 2 ** E_s, endpoint=False)
    main_bit = E_e
    possible_quantized = list()
    abs_max = torch.max(torch.abs(p.data)).item()
    upper = math.ceil(math.log2(abs_max / max(ratio) if E_e > 0 else max(ratio) - 1))
    lower = upper + 1 - 2 ** main_bit

    possible_main = list(map(lambda x: 2 ** x, range(int(lower), int(upper + 1))))

    for i in possible_main:
        for r in ratio:
            if len(possible_main) == 0 or i == min(possible_main):
                possible_quantized.append(2 * i * (r-1))
            else:
                possible_quantized.append(i * r)

    device = p.data.device
    possible_quantized = torch.tensor(possible_quantized).to(device)
    abs_p = torch.abs(p).view(-1)
    shape = p.shape
    idxs = (abs_p.unsqueeze(0) - possible_quantized.unsqueeze(1)).abs().min(dim=0)[1]
    quantized_p = possible_quantized.data[idxs]

    n_q = np.array([torch.sum((quantized_p == q).int()) for q in possible_quantized]) / p.numel()
    cum_q = np.cumsum(n_q)

    mn = 0
    mx = torch.max(p).item()
    hist, bin_edges = np.histogram(np.abs(p.detach().cpu().numpy()), bins='sqrt', range=(mn, mx), density=True)
    bin = [bin_edges[i] + bin_edges[i+1] for i in range(len(bin_edges) - 1)]
    interp_q = np.interp(bin, possible_quantized, cum_q)

    cum_p = np.cumsum(hist / np.sum(hist))

    KL = np.sum((cum_p - interp_q) * np.log2(cum_p / interp_q), where=cum_p * interp_q > 0) / np.sum(cum_p * interp_q > 0)
    return KL


def getQuanMSE(N_q, E_e, resume=None):
    """ Compute average KL-divergence of a model loaded from `resume`.

        :param N_q: total quantization bit-width
        :param E_e: mantissa bit-width
        :param resume: path to the model to be loaded
    """
    E_s = N_q - E_e - 1     # N_q: total bits, E_e: exponent bits
    root_path = './result/'
    ourMSE_list = list()
    if resume is None:
        resume = 'mobilenet_v2-b0353104.pth'
    baseline_path = root_path + resume
    assert os.path.isfile(baseline_path), 'Path is wrong!'
    if os.path.isfile(baseline_path):
        if 'state_dict' in torch.load(baseline_path).keys():
            baseline_checkpoint = torch.load(baseline_path)['state_dict']
        else:
            baseline_checkpoint = torch.load(baseline_path)

        for name, param in baseline_checkpoint.items():
            if param.shape.__len__() is 4:
                assert E_s >= 0, '>> Input the error format of E_s!, current {} = {} - {}'.format(E_s, N_q, E_e)

                base_data = param
                ourMSE = compute_KL(base_data, E_e, E_s)
                ourMSE_list.append(ourMSE)

    return np.average(ourMSE_list)


class SA(object):
    def __init__(self, interval, tab='min', T_max=100, T_min=1, iterMax=1, rate=0.8):
        print("================ current exponent is {} =================".format(exponent))
        self.interval = interval
        self.T_max = T_max
        self.T_min = T_min
        self.iterMax = iterMax
        self.rate = rate
        #############################################################
        self.x_seed = random.randint(2, interval[1])
        self.y_seed = random.randint(1, self.x_seed-1)
        self.x_seed = 8
        self.y_seed = 6
        self.x_list = []
        self.y_list = []
        self.converge = 0
        self.loss_list = []
        self.tab = tab.strip()
        #############################################################
        self.solve()
        self.display()

    def solve(self):
        temp = 'deal_' + self.tab
        if hasattr(self, temp):
            deal = getattr(self, temp)
        else:
            exit()
        x1 = np.round(self.x_seed)
        y1 = np.round(self.y_seed)
        interval = self.interval
        T = self.T_max
        # print('x1 >>: ', x1)
        # print('y1 >>: ', y1)
        while T >= self.T_min:
            print("\t++++++++++++ T = %f ++++++++++++" % T)
            for i in range(self.iterMax):
                x2 = random.randint(max(x1-1, y1+1, interval[0]), min(x1+1, interval[1]))
                y2 = random.randint(max(y1-1, 0), min(y1+1, x2-1, x1-1))

                f1 = self.func(x1, y1)
                self.x_list.append(x1)
                self.y_list.append(y1)
                self.loss_list.append(f1)
                f2 = self.func(x2, y2)
                delta_f = f2 - f1
                x1, converge_x = deal(x1, x2, delta_f, T)
                y1, converge_y = deal(y1, y2, delta_f, T)
                if converge_x and converge_y:
                    self.converge += 1
                else:
                    self.converge = 0
                print(x1, y1, self.converge)
                if self.converge >= 5:
                    T = 0
                    break

            T *= self.rate
        self.x_solu = x1
        self.y_solu = y1

    def func(self, x, y):
        value = penalty(y, x-y-1) * (getQuanMSE(x, y)*10)**4                    #x: N_q, y:E_e
        # print('This is Mse: {}'.format(value))
        return value

    def p_min(self, delta, T):
        probability = np.exp(-delta/T)
        return probability

    def p_max(self, delta, T):
        probability = np.exp(delta/T)
        return probability

    def deal_min(self, x1, x2, delta, T):
        if delta < 0:
            return x2, False
        else:
            P = self.p_min(delta, T)
            if P > random.random():
                return x2, False
            else:
                return x1, True

    def deal_max(self, x1, x2, delta, T):
        if delta > 0:
            return x2, False
        else:
            P = self.p_max(delta, T)
            if P > random.random():
                return x2, False
            else:
                return x1, True

    def display(self):
        pass
        print('solution: {}, {}'.format(self.x_solu, self.y_solu), file=file)
        print('solution: {}, {}'.format(self.x_solu, self.y_solu))
        print('x\n{}'.format(self.x_list), file=file)
        print('y\n{}'.format(self.y_list), file=file)
        print('loss\n{}'.format(self.loss_list), file=file)


def penalty(E_e, E_s):
    return 2 ** E_e + E_s + (E_e + ((1 if E_e > 0 else 0) + E_s) ** 2)


def function(N_q, E_e, exponent=0.5):
    return penalty(E_e, N_q-1-E_e) ** exponent * (getQuanMSE(N_q, E_e)*10)


def find_best(exponent=0.5):
    result = dict()
    for i in range(3, 9):
        for j in range(0, i):
            pass
            temp = function(i,j,exponent)
            # print(i, j, penalty(j, i-1-j), getQuanMSE(i, j), temp)
            result[(i, j)] = temp

    for k, v in result.items():
        if v == min(result.values()):
            print('best result is {}, {}'.format(k, v))
            return k


if __name__ == '__main__':
    for exponent in np.linspace(0, 1, 11):
        print("========  current exponent is {}  ========".format(exponent))
        find_best(exponent)
