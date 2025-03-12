"""
This code is adapted from PeSTo (Protein Structure Transformer).
Developed by the PeSTo authors, LBM, EPFL.
Original License: CC BY-NC-SA 4.0
Copyright (c) original source: https://github.com/LBM-EPFL/PeSTo/tree/main
"""

import torch as pt
from utils.model_operations import StateUpdateLayer, StatePoolLayer, unpack_state_features, normalize_sasa


class Model(pt.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.em = pt.nn.Sequential(
            pt.nn.Linear(config['em']['N0'], config['em']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['em']['N1'], config['em']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['em']['N1'], config['em']['N1']),
        )
        # atomic level state update model
        self.sum = pt.nn.Sequential(*[StateUpdateLayer(layer_params) for layer_params in config['sum']])
        
        # atomic to residue reduction layer
        self.spl = StatePoolLayer(config['spl']['N0'], config['spl']['N1'], config['spl']['Nh'])

        # decoding mlp
        self.dm = pt.nn.Sequential(
            pt.nn.Linear(3*config['dm']['N0'], 2*config['dm']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(2*config['dm']['N1'], config['dm']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['dm']['N1'], config['dm']['N2']),
        )

    def forward(self, onehot_seq, rmsf1, rmsf2, rsa, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, mapping):
        # concat and encode node features
        normalized_rsa = normalize_sasa(rsa).to(rsa.device)
        q0 = pt.cat((onehot_seq, normalized_rsa, rmsf1, rmsf2),dim=1).float() #30+3
        q = self.em.forward(q0)
        
        # initial state vectors
        p_r = pt.zeros((q.shape[0]+1, 3, q.shape[1]), device=q.device) #[N, 3, S]
        p_motion = pt.zeros((q.shape[0]+1, 3, q.shape[1]), device=q.device) #[N, 3, S]

        # unpack state features with sink
        q, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, ntype = unpack_state_features(q, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, rsa)
       
        # atomic tsa layers
        qa, pa_r, pa_motion,_,_,_,_,_,_,_,in_column,out_column = self.sum.forward((q, p_r, p_motion, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, ntype,0,0))

        # decode state (without sink)
        qr, pr_r, pr_motion = self.spl.forward(qa[1:], pa_r[1:], pa_motion[1:], mapping)
        zr = pt.cat([qr, pt.norm(pr_r, dim=1), pt.norm(pr_motion, dim=1)], dim=1)
        z = self.dm.forward(zr)
        return z,in_column,out_column
