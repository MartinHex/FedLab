# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class Aggregators(object):
    """Define the algorithm of parameters aggregation"""

    @staticmethod
    def fedavg_aggregate(serialized_params_list, weights=None, mean_shift = False,client_shift=False):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)

        assert torch.all(weights >= 0), "weights should be non-negative values"

        serialized_params_stack = torch.stack(serialized_params_list, dim=-1)

        if client_shift:
            serialized_parameters = 0
            for serialized_client_params,w in zip(serialized_params_list,weights):
                serialized_parameters+=mean_shift(serialized_params_stack,serialized_client_params)*w
        else:
            serialized_parameters = torch.sum(
                serialized_params_stack * weights, dim=-1)

        if mean_shift:
            serialized_parameters = _mean_shift(serialized_params_stack, serialized_parameters )
           
           
        return serialized_parameters 


    @staticmethod
    def fedasync_aggregate(server_param, new_param, alpha):
        """FedAsync aggregator
        
        Paper: https://arxiv.org/abs/1903.03934
        """
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, new_param)
        return serialized_parameters

def _mean_shift(client_weights,init,tol=1e-6,device=None):
    client_weights = torch.transpose(client_weights, 0, 1)
    # print(f'init shape {init.shape}')
    # print(f'weigth shape {client_weights.shape}')
    max_iter = 10
    bandwiths = _silverman(client_weights)
    w_res = init
    non_fixed_idx = torch.std(client_weights,0).nonzero().flatten()
    for i in range(max_iter): # max_iter
        # print(i)
        # print(f'non zero parameters: {len(non_fixed_idx)}')
        # Initiate parameters which are to be mean-shifted
        w = w_res[non_fixed_idx]
        # print('a')
        n_nonzeros = len(w)
        if n_nonzeros==0: break
        denominator= torch.zeros(n_nonzeros)#.to(device)
        numerator = torch.zeros(n_nonzeros)#.to(device)
        # print('b')
        H = bandwiths[non_fixed_idx]
        # print('c')
        for _,client_w in enumerate(client_weights):
            w_i = client_w[non_fixed_idx]#.to(device)
            K = _epanachnikov_kernel((w-w_i)/H)
            denominator += K
            numerator += K*w_i
        # print('d')
        # Calculate the mean shift
        m_x = numerator/denominator
        # Replace nan values with previus iteration
        nan_idx = m_x.isnan().nonzero().flatten()
        m_x[nan_idx] = w[nan_idx]
        # Update resulting parameters
        w_res[non_fixed_idx] = m_x
        # Update which parameters which are to be selected for next iteration
        non_converged_idx = torch.abs(w-m_x)>tol
        non_fixed_idx = non_fixed_idx[non_converged_idx]
    return w_res

def _gaussian_kernel(u):
    return torch.exp(-u**2/2)

def _epanachnikov_kernel(u):
    u[torch.abs(u)>1] = 1
    return 3/4 * (1 - u**2)

def _silverman(client_weights):
    n = len(client_weights)
    h = torch.std(client_weights,0)
    iqr = torch.quantile(client_weights,0.75,dim=0)-torch.quantile(client_weights,0.25,dim=0)
    h[h>iqr] = iqr[h>iqr]
    return (n*3/4)**(-0.2)*h

