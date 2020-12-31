import torch


dtype = None
device = None
n_steps = None
syn_a = None
tau_s = None

def init(dty, dev, params):   # a(t_k) = (1/tau)exp(-(t_k-t_m)/tau)H(t_k-t_m)
    global dtype, device, n_steps, syn_a, partial_a, tau_s, tau_m,\
    layers_name, grad_norm_dict, last_layer_name

    dtype = dty
    device = dev
    n_steps = params['Network']['n_steps']
    tau_s = params['Network']['tau_s']
    tau_m = params['Network']['tau_m']
    layers_name = []

    grad_norm_dict = {}
    layers_config = params['Layers']
    for key in layers_config:
        layers_name += [key]
    syn_a = torch.zeros((1, 1, 1, 1, n_steps), dtype=dtype, device=device)
    syn_a[..., 0] = 1
    for t in range(n_steps-1):
        syn_a[..., t+1] = syn_a[..., t] - syn_a[..., t] / tau_s
    syn_a /= tau_s
