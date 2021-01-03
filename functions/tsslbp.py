import torch
import torch.nn as nn
import torch.nn.functional as f
from time import time
import global_v as glv
import neighbors as nb

class TSSLBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config, name):
        shape = inputs.shape
        n_steps = shape[4]
        theta_m = 1/network_config['tau_m']
        tau_s = network_config['tau_s']
        theta_s = 1/tau_s
        threshold = layer_config['threshold']

        mem = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
        syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype).to(glv.device)
        syns_posts = []
        mems = []
        mem_updates = []
        outputs = []
        for t in range(n_steps):
            mem_update = (-theta_m) * mem + inputs[..., t]
            mem += mem_update

            out = mem > threshold
            out = out.type(glv.dtype)
            mems.append(mem)

            mem = mem * (1-out)
            outputs.append(out)
            mem_updates.append(mem_update)
            syn = syn + (out - syn) * theta_s
            syns_posts.append(syn)

        mems = torch.stack(mems, dim = 4)
        mem_updates = torch.stack(mem_updates, dim = 4)
        outputs = torch.stack(outputs, dim = 4)
        syns_posts = torch.stack(syns_posts, dim = 4)
        layer_index = glv.layers_name.index(name)
        ctx.save_for_backward(mem_updates, outputs, mems, syns_posts,\
                torch.tensor([threshold, tau_s, theta_m, layer_index]))

        return syns_posts

    @staticmethod
    def backward(ctx, grad_delta):
        (delta_u, outputs, u, syns_posts, others) = ctx.saved_tensors
        shape = grad_delta.shape
        neuron_num = shape[0] * shape[1] * shape[2] * shape[3]
        n_steps = shape[4]
        threshold = others[0].item()
        tau_s = others[1].item()
        theta_m = others[2].item()
        name = others[3].item()

        projects = nb.get_projects(outputs, u, name, syns_posts, grad_delta)
        projects = projects.T.view(shape)
        dist_aggregate_factor = 0.2/((u-threshold)**2 + 0.2)
        #m = torch.nn.Softmax(dim=-1)
        #dist_aggregate_factor = m(dist_aggregate_factor)
        #grad = grad_delta * dist_aggregate_factor
        grad = projects * (outputs - 0.5) * 2 * dist_aggregate_factor#sig_grad
        nb.update_norm(grad, name)
        mean = torch.mean(torch.abs(grad))
        last_norm = glv.grad_norm_dict[glv.last_layer_name]
        #grad = grad/mean * last_norm * torch.log(mean/last_norm + 1.1)
        grad = grad * torch.log(last_norm/mean + 1.02) * 1.2
        nb.update_norm(grad, name)
        """

        grad = torch.zeros_like(grad_delta)
        syn_a = glv.syn_a.repeat(shape[0], shape[1], shape[2], shape[3], 1)
        for t in range(n_steps):
            # time_end = int(min(t+tau_s, n_steps))
            time_end = n_steps
            time_len = time_end-t
            grad_a = torch.sum(syn_a[..., 0:time_len]*grad_delta[..., t:time_end], dim=-1)

            a = 0.2
            f = torch.clamp((-1 * u[..., t] + threshold) / a, -8, 8)
            f = torch.exp(f)
            f = f / ((1 + f) * (1 + f) * a)

            grad[..., t] = grad_a * f
        #grad = grad + grad_n
        """
        return grad, None, None, None
