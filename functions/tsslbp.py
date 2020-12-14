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
        neighbors = nb.neighbors_predict(outputs, u, name)
        neighbors_syns_posts = nb.neighbors_syns_posts(neighbors)
        projects = nb.projects(neighbors_syns_posts, syns_posts, grad_delta)
        best_neighbor = torch.argmax(projects, dim=0)
        #neighbors = neighbors.view(neuron_num * n_steps, -1)
        #best_neighbor = best_neighbor * neuron_num +\
        #    torch.tensor(range(neuron_num))
        #selected_neighbor = neighbors[best_neighbor]
        mask = torch.eye(n_steps)[best_neighbor].view(shape)
        mask = mask * (outputs - 0.5) * 0.2
        projects = projects.view(n_steps * neuron_num, -1)
        best_projects = projects[best_neighbor * neuron_num +\
                torch.tensor(range(neuron_num))]
        #best_projects = best_projects.where(best_projects > 0, best_projects * 0)
        best_projects = best_projects.repeat(1, n_steps).view(shape)
        #sig = nb.sigmoid(u, 0.2)
        #sig_grad = sig * (1 - sig) / 0.2
        grad = mask * best_projects# * grad_delta * sig_grad
        # print(best_neighbor.shape)
        # print(mask.shape)
        # print(projects.shape)
        # print(neighbors.shape)
        # print(best_projects[0,0,0,0])
        #grad = None
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
        grad = grad + grad_n
        """
        return grad, None, None, None