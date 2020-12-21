import torch
import global_v as glv

def neighbors_predict(outputs, u, k):
    tau_m = glv.tau_m
    m_decay = (1 - 1 / tau_m)
    shape = outputs.shape
    time_steps = glv.n_steps
    neuron_num = shape[0] * shape[1] * shape[2] * shape[3]
    outputs = outputs.view(neuron_num, shape[4])
    u = u.view(neuron_num, shape[4])
    threshold = 1
    neighbors = []
    for t in range(time_steps):
        neighbor_output = outputs > 0
        near_by = torch.ones(neuron_num, dtype=torch.bool, device=glv.device)
        current_t = t
        while near_by.any() == True:
            neighbor_output[:, current_t] = neighbor_output[:, current_t] ^ near_by
            current_t += 1
            if current_t == time_steps:
                break
            # New output of the previous time step
            nopp = neighbor_output[:, current_t - 1]
            # Membrane potential of the current time step
            mbp = u[:, current_t]
            near_by = near_by&\
                ((near_by&(nopp==True)&((threshold<=mbp)&(mbp<(threshold+m_decay))))|\
                (near_by&(nopp==False)&(((threshold-m_decay)<mbp)&(mbp<threshold))))
        neighbors += [neighbor_output]
    return torch.stack(neighbors, dim=0)

def neighbors_syns_posts(neighbors):
    tau_s = glv.tau_s
    theta_s = 1/tau_s
    shape = neighbors.shape
    neighbors = neighbors.type(glv.dtype)
    neighbors = neighbors.view(shape[0]*shape[1], shape[2])
    syns_posts = []
    syn = torch.zeros(shape[0]*shape[1], dtype=glv.dtype, device=glv.device)
    for i in range(shape[2]):
        syn = syn + (neighbors[:, i] - syn) * theta_s
        syns_posts.append(syn)
    syns_posts = torch.stack(syns_posts, dim = 1)
    syns_posts = syns_posts.view(shape[0], shape[1], shape[2])
    return syns_posts

def projects(neighbors_syns_posts, syns_posts, grad_delta):
    shape = syns_posts.shape
    syns_posts = syns_posts.reshape(shape[0]*shape[1]*shape[2]*shape[3], shape[4])
    grad_delta = grad_delta.reshape(shape[0]*shape[1]*shape[2]*shape[3], shape[4])
    syns_posts = syns_posts.repeat(shape[4], 1, 1)
    grad_delta = grad_delta.repeat(shape[4], 1, 1)
    delta_syns_posts = neighbors_syns_posts - syns_posts
    dot_product = torch.sum(delta_syns_posts * (- grad_delta), dim = -1)
    d_syns_norm = torch.sqrt(torch.sum(delta_syns_posts * delta_syns_posts,\
            dim = -1))
    grad_d_norm = torch.sqrt(torch.sum(grad_delta * grad_delta, dim=-1))
    projects = dot_product/(d_syns_norm * grad_d_norm)
    return projects

def sigmoid(x, temp):
    exp = torch.clamp(-x/temp, -10, 10)
    return 1 / (1 + torch.exp(exp))
