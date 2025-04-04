"""
This code is adapted from PeSTo (Protein Structure Transformer).
Developed by the PeSTo authors, LBM, EPFL.
Original License: CC BY-NC-SA 4.0
Copyright (c) original source: https://github.com/LBM-EPFL/PeSTo/tree/main
"""

import torch as pt
from torch.utils.checkpoint import checkpoint
from utils.configs import config_runtime
from model_extras import normalize_sasa, ActionUpdate, print_free_gpu_memory

                          
def unpack_state_features(q, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, rsa):
    # mask distances (to exclude edge with itself)
    D_nn = D_nn + pt.max(D_nn)*(D_nn < 1e-2).float()
    # make node type vector (-1 & 1)
    ntype = (rsa >= config_runtime['rsa_thr']).float() * 2 - 1
    
    # prepare sink
    q = pt.cat([pt.zeros((1, q.shape[1]), device=q.device), q], dim=0)
    nn_topk = pt.cat([pt.zeros((1, nn_topk.shape[1]), dtype=pt.long, device=nn_topk.device), nn_topk], dim=0)
    D_nn = pt.cat([pt.zeros((1, D_nn.shape[1], D_nn.shape[2]), device=D_nn.device), D_nn], dim=0)
    R_nn = pt.cat([pt.zeros((1, R_nn.shape[1], R_nn.shape[2]), device=R_nn.device), R_nn], dim=0)
    motion_v_nn = pt.cat([pt.zeros((1, motion_v_nn.shape[1], motion_v_nn.shape[2]), device=motion_v_nn.device), motion_v_nn], dim=0)
    motion_s_nn = pt.cat([pt.zeros((1, motion_s_nn.shape[1], motion_s_nn.shape[2]), device=motion_s_nn.device), motion_s_nn], dim=0)
    CP_nn = pt.cat([pt.zeros((1, CP_nn.shape[1], CP_nn.shape[2]), device=CP_nn.device), CP_nn], dim=0)
    ntype = pt.cat([pt.tensor([0], dtype=ntype.dtype, device=q.device), ntype.squeeze(1)], dim=0)
    
    return q, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, ntype

# >>> OPERATION        
class StateUpdate(pt.nn.Module):
    def __init__(self, Ns, Nh, Nk):
        super(StateUpdate, self).__init__()
        # operation parameters
        self.Ns = Ns
        self.Nh = Nh
        self.Nk = Nk

        # node query model
        self.nqm = pt.nn.Sequential(
            pt.nn.Linear(3*Ns, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 2*Nk*Nh),
        )

        # edges scalar keys model
        self.eqkm = pt.nn.Sequential(
            pt.nn.Linear(10*Ns+3, 4*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(4*Ns, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, Nk),
        )

        # edges vector keys model
        self.epkm = pt.nn.Sequential(
            pt.nn.Linear(10*Ns+3, 8*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(8*Ns, 4*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(4*Ns, 3*Nk),
        )

        # edges value model
        self.evm = pt.nn.Sequential(
            pt.nn.Linear(10*Ns+3, 8*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(8*Ns, 3*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(3*Ns, 3*Ns),
        )

        # scalar projection model
        self.qpm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

        # vector projection model
        self.ppm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns, bias=False),
        )

        # scaling factor for attention
        self.sdk = pt.nn.Parameter(pt.sqrt(pt.tensor(Nk).float()), requires_grad=False)

    def forward(self, q, p_r, p_motion, q_nn, p_r_nn, p_motion_nn, d_nn, r_nn, motion_v_nn, motion_s_nn, CP_nn, mask):
        # q: [N, S]
        # p: [N, 3, S]
        # q_nn: [N, n, S]
        # p_nn: [N, n, 3, S]
        # d_nn: [N, n]
        # r_nn: [N, n, 3]
        # dc_nn: [N, n]
        # mask: [N, N]
        # N: number of nodes
        # n: number of nearest neighbors
        # S: state dimensions
        # H: number of attention heads

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
            X_n.unsqueeze(1).repeat(1,n,1),                                   # centered state (N, n, 3*S)
            q_nn,                                                             # neighbors scalar states (N, n, S)
            pt.norm(p_r_nn, dim=2),                                           # neighbors vector states r norms (N, n, S)
            pt.norm(p_motion_nn, dim=2),                                      # neighbors vector states motion norms (N, n, S)
            pt.sum(p_r.unsqueeze(1) * r_nn.unsqueeze(3), dim=2),              # r - centered vector state projections (N, n, S)
            pt.sum(p_r_nn * r_nn.unsqueeze(3), dim=2),                        # r - neighbors vector states projections (N, n, S)
            pt.sum(p_motion.unsqueeze(1) * motion_v_nn.unsqueeze(3), dim=2),  # dynamic motion - centered vector state projections (N, n, S)
            pt.sum(p_motion_nn * motion_v_nn.unsqueeze(3), dim=2),            # dynamic motion - neighbors vector states projections (N, n, S)
            motion_s_nn,                                                      # scalar dynamic motion (N, n, 1)
            CP_nn,                                                            # communication propensity (N, n, 1)
        ], dim=2)  # [N, n, 10*S+3]

        # node queries
        Q = self.nqm.forward(X_n).view(N, 2, self.Nh, self.Nk)  # [N, 2, Nh, Nk]

        # scalar edges keys while keeping interaction order inveriance
        Kq = self.eqkm.forward(X_e).view(N, n, self.Nk).transpose(1,2)  # N, Nk, n]

        # vector edges keys while keeping bond order inveriance
        Kp = pt.cat(pt.split(self.epkm.forward(X_e), self.Nk, dim=2), dim=1).transpose(1,2)  # [N, Nk, 3*n]

        # edges values while keeping interaction order inveriance
        V = self.evm.forward(X_e).view(N, n, 3, S).transpose(1,2)  # [N, 3, n, S]
        
        # vectorial inputs packings
        Vp_r = pt.cat([
            V[:,1].unsqueeze(2) * r_nn.unsqueeze(3),   #[N ,n ,3 ,S]
            p_r.unsqueeze(1).repeat(1,n,1,1),          #[N ,n ,3, S]
            p_r_nn,                                    #[N ,n ,3, S]
        ], dim=1).transpose(1,2)  # [N, 3, 3*n, S]
        Vp_motion = pt.cat([
            V[:,2].unsqueeze(2) * motion_v_nn.unsqueeze(3),   #[N ,n ,3 ,S]
            p_motion.unsqueeze(1).repeat(1,n,1,1),     #[N ,n ,3, S]
            p_motion_nn,                               #[N ,n ,3, S]
        ], dim=1).transpose(1,2)  # [N, 3, 3*n, S]
        
        # queries and keys collapse
        QKq = pt.matmul(Q[:,0], Kq) / self.sdk # [N, Nh, n]
        QKp = pt.matmul(Q[:,1], Kp) / self.sdk # [N, Nh, 3*n]
        
        # mask the action
        QKq_masked = QKq * mask.unsqueeze(1)
        QKp_masked = QKp * mask.unsqueeze(1).repeat(1, 1, 3).reshape(mask.shape[0],1,-1)
        
        # apply softmax
        Mq = pt.nn.functional.softmax(QKq_masked, dim=2)  # [N, Nh, n]
        Mp = pt.nn.functional.softmax(QKp_masked, dim=2)  # [N, Nh, 3*n]

        # attention mask and values collapse
        Zq = pt.matmul(Mq, V[:,0]).view(N, self.Nh*self.Ns)  # [N, Nh*S]
        Zp_r = pt.matmul(Mp.unsqueeze(1), Vp_r).view(N, 3, self.Nh*self.Ns)  # [N, 3, Nh*S]
        Zp_motion = pt.matmul(Mp.unsqueeze(1), Vp_motion).view(N, 3, self.Nh*self.Ns)  # [N, 3, Nh*S]

        # decode outputs
        qh = self.qpm.forward(Zq)
        ph_r = self.ppm.forward(Zp_r)
        ph_motion = self.ppm.forward(Zp_motion)

        # update state with residual
        qz = q + qh
        pz_r = p_r + ph_r
        pz_motion = p_motion + ph_motion

        return qz, pz_r, pz_motion

    
# >>> LAYER
class StatePoolLayer(pt.nn.Module):
    def __init__(self, N0, N1, Nh):
        super(StatePoolLayer, self).__init__()
        # state attention model
        self.sam = pt.nn.Sequential(
            pt.nn.Linear(3*N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, 2*Nh),
        )

        # attention heads decoding
        self.zdm = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N1),
        )

        # vector attention heads decoding
        self.zdm_vec = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N1, bias=False)
        )

    def forward(self, q, p_r, p_motion, mapping):
        M = pt.zeros((len(mapping), mapping[-1]), dtype=pt.int).to(q.device)
        indices = (mapping - 1).unsqueeze(1).to(q.device)
        M.scatter_(1, indices, 1)
    
        # create filter for softmax
        F = (1.0 - M + 1e-6) / (M - 1e-6)

        # pack features
        z = pt.cat([q, pt.norm(p_r, dim=1), pt.norm(p_motion, dim=1)], dim=1)

        # multiple attention pool on state
        Ms = pt.nn.functional.softmax(self.sam.forward(z).unsqueeze(1) + F.unsqueeze(2), dim=0).view(M.shape[0], M.shape[1], -1, 2)
        qh = pt.matmul(pt.transpose(q,0,1), pt.transpose(Ms[:,:,:,0],0,1))
        ph_r = pt.matmul(pt.transpose(pt.transpose(p_r,0,2),0,1), pt.transpose(Ms[:,:,:,1],0,1).unsqueeze(1))
        ph_motion = pt.matmul(pt.transpose(pt.transpose(p_motion,0,2),0,1), pt.transpose(Ms[:,:,:,1],0,1).unsqueeze(1))

        # attention heads decoding
        qr = self.zdm.forward(qh.view(Ms.shape[1], -1))
        pr_r = self.zdm_vec.forward(ph_r.view(Ms.shape[1], ph_r.shape[1], -1))
        pr_motion = self.zdm_vec.forward(ph_motion.view(Ms.shape[1], ph_motion.shape[1], -1))
        return qr, pr_r, pr_motion
        
class StateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(StateUpdateLayer, self).__init__()
        # define operations
        self.in_au = ActionUpdate(*[layer_params[k] for k in ['Ns']])
        self.out_au = ActionUpdate(*[layer_params[k] for k in ['Ns']])
        self.su = StateUpdate(*[layer_params[k] for k in ['Ns', 'Nh', 'Nk']])
        # store number of nearest neighbors
        self.m_nn = pt.nn.Parameter(pt.arange(layer_params['nn'], dtype=pt.int64), requires_grad=False)

    def forward(self, z):
        q, p_r, p_motion, nn_topk, D_topk, R_topk, motion_v_nn, motion_s_nn, CP_nn, ntype,_,_ = z
        ids_nn = nn_topk[:,self.m_nn]to(pt.int64)
        
        #print_free_gpu_memory()
        # Compute action update logits
        in_logits = self.in_au(
            q, p_r, p_motion, 
            q[ids_nn], p_r[ids_nn], p_motion[ids_nn], 
            D_topk[:, self.neighbor_indices], 
            R_topk[:, self.neighbor_indices], 
            motion_v_nn[:, self.neighbor_indices], 
            motion_s_nn[:, self.neighbor_indices], 
            CP_nn[:, self.neighbor_indices]
        )
        out_logits = self.out_au(
            q, p_r, p_motion, 
            q[ids_nn], p_r[ids_nn], p_motion[ids_nn], 
            D_topk[:, self.neighbor_indices], 
            R_topk[:, self.neighbor_indices], 
            motion_v_nn[:, self.neighbor_indices], 
            motion_s_nn[:, self.neighbor_indices], 
            CP_nn[:, self.neighbor_indices]
        )
        
        # make the mask matrix
        in_column = in_logits[:, 0].unsqueeze(1)
        out_column = out_logits[:, 0].unsqueeze(0)
        mask = in_column * out_column
        ntype_mask = ntype.unsqueeze(1) * ntype.unsqueeze(0)
        mask[(mask == 0) & (ntype_mask == 1)] = 1 # neutralizing mask for intra-type nodes

        # update q and ps with checkpoint
        q = q.detach().requires_grad_()
        p_r = p_r.detach().requires_grad_()
        p_motion = p_motion.detach().requires_grad_()
        q, p_r, p_motion = checkpoint(
            self.su,
            q, p_r, p_motion,
            q[ids_nn], p_r[ids_nn], p_motion[ids_nn],
            D_topk[:, self.neighbor_indices],
            R_topk[:, self.neighbor_indices],
            motion_v_nn[:, self.neighbor_indices],
            motion_s_nn[:, self.neighbor_indices],
            CP_nn[:, self.neighbor_indices],
            mask[:, self.neighbor_indices],
            use_reentrant=False
        )
        # Zero out the first node's states
        with pt.no_grad():
            q[0].zero_()
            p_r[0].zero_()
            p_motion[0].zero_()

        return q, p_r, p_motion, nn_topk, D_topk, R_topk, motion_v_nn, motion_s_nn, CP_nn, ntype, in_column, out_column

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
