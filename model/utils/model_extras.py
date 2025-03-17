"""
model_extras.py: Additional operations
Omid Mokhtari - Inria 2025
This file is part of DynamicGT.
Released under CC BY-NC-SA 4.0 License
"""

import torch as pt
import subprocess


def print_free_gpu_memory():
    try: # query the GPU memory usage 
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], encoding='utf-8')
        free_memory = int(result.split('\n')[0])
        print(f"Free GPU Memory before layer {next(iterator_marks)+1}: {free_memory} MiB")
        print ("\n\n\n")
    except Exception as e:
        print("Could not query GPU memory usage:", e)


def normalize_sasa(sasa):
    min_sasa = min(sasa)
    max_sasa = max(sasa)
    range_sasa = max_sasa - min_sasa
    if range_sasa == 0:
        print ('range = 0')
        return pt.tensor([0] * len(sasa)).view(-1,1)
    sasa = [(value - min_sasa)/ range_sasa for value in sasa]
    return pt.tensor(sasa).view(-1,1)
    

class ActionUpdate(pt.nn.Module):
    def __init__(self, Ns):
        super(ActionUpdate, self).__init__()
        self.Ns = Ns
        # message computation and aggregation
        self.message_func = pt.nn.Sequential(
            pt.nn.Linear(10*Ns+3, 8*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(8*Ns, 4*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(4*Ns, 4*Ns))
        # message transformation to binary logits
        self.node_transform = pt.nn.Sequential(
            pt.nn.Linear(7*Ns, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 2))
        
        # Intitializing temp parameter
        self.temperature = pt.nn.Parameter(pt.tensor([0.5]))
        
    def forward(self, q, p_r, p_motion, q_nn, p_r_nn, p_motion_nn, d_nn, r_nn, motion_v_nn, motion_s_nn, CP_nn):
    
        # get dimensions
        N, n, S = q_nn.shape
    
        # node inputs packing
        X_n = pt.cat([
            q,
            pt.norm(p_r, dim=1),
            pt.norm(p_motion, dim=1),
        ], dim=1)  # [N, 3*S]

        # edge inputs packing
        X_e = pt.cat([
            d_nn,                                                             # distance (N, n, 1)
            X_n.unsqueeze(1).repeat(1,n,1),                                   # centered node state (N, n, 3*S)
            q_nn,                                                             # neighbors scalar states (N, n, S)
            pt.norm(p_r_nn, dim=2),                                           # neighbors vector states r norms (N, n, S)
            pt.norm(p_motion_nn, dim=2),                                      # neighbors vector states motion norms (N, n, S)
            pt.sum(p_r.unsqueeze(1) * r_nn.unsqueeze(3), dim=2),              # r - centered vector state projections (N, n, S)
            pt.sum(p_r_nn * r_nn.unsqueeze(3), dim=2),                        # r - neighbors vector states projections (N, n, S)
            pt.sum(p_motion.unsqueeze(1) * motion_v_nn.unsqueeze(3), dim=2),  # dynamic motion - centered vector state projections (N, n, S)
            pt.sum(p_motion_nn * motion_v_nn.unsqueeze(3), dim=2),            # dynamic motion - neighbors vector states projections (N, n, S)
            motion_s_nn,                                                      # scalar dynamic motion (N, n, 1)
            CP_nn,                                                            # Communication Propensity (N, n, 1)
        ], dim=2)  # [N, n, 10*S+3]
        
        # Message calculation and aggregation
        messages = self.message_func.forward(X_e)               # [N, n, 4*S]
        aggregated_message = pt.mean(messages, dim=1)           # [N, 4*S]
        # concatenate node with edge message
        X_ne = pt.cat([aggregated_message, X_n], dim=1)         # [N, 7*S]
        # final transformation
        logits = self.node_transform(X_ne)                   # [N, 2]
        # gumbel_softmax
        temp = pt.clamp(self.temperature, min=0.1)
        act = pt.nn.functional.gumbel_softmax(logits, tau=temp, hard=True)
        
        return act
