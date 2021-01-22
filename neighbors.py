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
    #grad_d_norm = torch.sqrt(torch.sum(grad_delta * grad_delta, dim=-1))
    projects = dot_product/d_syns_norm# * grad_d_norm)
    return projects

def sigmoid(x, temp):
    exp = torch.clamp(-x/temp, -10, 10)
    return 1 / (1 + torch.exp(exp))

def get_projects(outputs, u, name, syns_posts, grad_delta):
    tau_m = glv.tau_m
    m_decay = (1 - 1 / tau_m)
    tau_s = glv.tau_s
    theta_s = 1/tau_s
    shape = outputs.shape
    time_steps = glv.n_steps
    neuron_num = shape[0] * shape[1] * shape[2] * shape[3]
    outputs = outputs.view(neuron_num, shape[4])
    u = u.view(neuron_num, shape[4])
    threshold = 1
    projects = []
    for t in range(time_steps):
        neighbor_output = outputs > 0
        S0 = (neighbor_output[:, t]).float()
        flip = torch.ones(neuron_num, dtype=torch.bool, device=glv.device)
        neighbor_output[:, t] = neighbor_output[:, t] ^ flip
        U_2 = u[:, t]
        u[:, t] = 1
        current_t = t+1
        if current_t == time_steps:
            flip[:] = False
        while flip.any() == True:
            U_1 = u[:, current_t]
            u[:,current_t] = u[:,current_t] + S0 * u[:, current_t-1] * m_decay\
                    + (1-S0) * (-U_2) * m_decay
            U_2 = U_1
            flip = flip & ((u[:,current_t]>1)^neighbor_output[:,current_t])
            neighbor_output[:, current_t] = neighbor_output[:, current_t] ^ flip
            current_t += 1
            if current_t == time_steps:
                flip[:] = False
        neighbor_output = neighbor_output.type(glv.dtype)
        neighbor_syns_posts = []
        syn = torch.zeros(neuron_num, dtype=glv.dtype, device=glv.device)
        for i in range(time_steps):
            syn = syn + (neighbor_output[:, i] - syn) * theta_s
            neighbor_syns_posts.append(syn)
        neighbor_syns_posts = torch.stack(neighbor_syns_posts, dim = 1)
        syns_posts = syns_posts.reshape(neuron_num, time_steps)
        grad_delta = grad_delta.reshape(neuron_num, time_steps)
        delta_syns_posts = neighbor_syns_posts - syns_posts
        dot_product = torch.sum(delta_syns_posts * (- grad_delta), dim = -1)
        d_syns_norm = torch.sqrt(torch.sum(delta_syns_posts * delta_syns_posts,\
            dim = -1))
        projects += [dot_product/(d_syns_norm+0.00001)]# * grad_d_norm 
    projects = torch.stack(projects, dim=0)
    return projects

def get_projects_simplified(outputs, u, name, syns_posts, grad_delta):
    tau_m = glv.tau_m
    m_decay = (1 - 1 / tau_m)
    tau_s = glv.tau_s
    theta_s = 1/tau_s
    shape = outputs.shape
    time_steps = glv.n_steps
    neuron_num = shape[0] * shape[1] * shape[2] * shape[3]
    outputs = outputs.view(neuron_num, shape[4])
    u = u.view(neuron_num, shape[4])
    threshold = 1
    neighbors = []
    projects = []
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
        neighbor_output = neighbor_output.type(glv.dtype)
        neighbor_syns_posts = []
        syn = torch.zeros(neuron_num, dtype=glv.dtype, device=glv.device)
        for i in range(time_steps):
            syn = syn + (neighbor_output[:, i] - syn) * theta_s
            neighbor_syns_posts.append(syn)
        neighbor_syns_posts = torch.stack(neighbor_syns_posts, dim = 1)
        syns_posts = syns_posts.reshape(neuron_num, time_steps)
        grad_delta = grad_delta.reshape(neuron_num, time_steps)
        delta_syns_posts = neighbor_syns_posts - syns_posts
        dot_product = torch.sum(delta_syns_posts * (- grad_delta), dim = -1)
        d_syns_norm = torch.sqrt(torch.sum(delta_syns_posts * delta_syns_posts,\
            dim = -1))
        projects += [dot_product/(d_syns_norm+0.00001)]# * grad_d_norm 
    projects = torch.stack(projects, dim=0)
    return projects

def get_projects_discrete(outputs, u, name, syns_posts, grad_delta):
    tau_m = glv.tau_m
    m_decay = (1 - 1 / tau_m)
    tau_s = glv.tau_s
    theta_s = 1/tau_s
    shape = outputs.shape
    time_steps = glv.n_steps
    neuron_num = shape[0] * shape[1] * shape[2] * shape[3]
    outputs = outputs.view(neuron_num, shape[4])
    u = u.view(neuron_num, shape[4])
    threshold = 1
    neighbors = []
    projects = []
    for t in range(time_steps):
        neighbor_output = outputs > 0
        flip = torch.ones(neuron_num, dtype=torch.bool, device=glv.device)
        if t != time_steps-1:
            next_flip = torch.ones(neuron_num, dtype=torch.bool, device=glv.device)
            next_flip = next_flip & (neighbor_output[:, t] != neighbor_output[:, t+1])
            next_flip = next_flip & (torch.abs(u[:, t]-1) > \
                    0.5*(torch.abs(u[:, t]-1)+torch.abs(u[:, t+1]-1)))
            neighbor_output[:, t+1] = neighbor_output[:, t+1] ^ next_flip
        neighbor_output[:, t] = neighbor_output[:, t] ^ flip
        neighbor_output = neighbor_output.type(glv.dtype)
        neighbor_syns_posts = []
        syn = torch.zeros(neuron_num, dtype=glv.dtype, device=glv.device)
        for i in range(time_steps):
            syn = syn + (neighbor_output[:, i] - syn) * theta_s
            neighbor_syns_posts.append(syn)
        neighbor_syns_posts = torch.stack(neighbor_syns_posts, dim = 1)
        syns_posts = syns_posts.reshape(neuron_num, time_steps)
        grad_delta = grad_delta.reshape(neuron_num, time_steps)
        delta_syns_posts = neighbor_syns_posts - syns_posts
        dot_product = torch.sum(delta_syns_posts * (- grad_delta), dim = -1)
        d_syns_norm = torch.sqrt(torch.sum(delta_syns_posts * delta_syns_posts,\
            dim = -1))
        projects += [dot_product/(d_syns_norm+0.00001)]# * grad_d_norm 
    projects = torch.stack(projects, dim=0)
    return projects
        #neighbors += [neighbor_output]
def index_to_spike_train(index, time_steps):
    return np.array(
            ['0']*(
                time_steps-len(list(bin(index)[2:]))
                )+list(bin(index)[2:])
            ).astype('int')
def spike_train_to_index(spike_train):
    return int(
            '0b'+''.join(
                str(
                    np.array(spike_train).astype('int')
                )[1:-1].split(' ')
            ),0)
def update_norm(grad, name):
    if len(glv.grad_norm_dict)==0:
        glv.last_layer_name = int(name)
    glv.grad_norm_dict[int(name)]\
    =round(torch.mean(torch.abs(grad)).cpu().item(), 3)
