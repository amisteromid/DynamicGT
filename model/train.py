from utils.configs import config_data, config_model, config_runtime
from utils.model import Model
from utils.data_handler import Dataset, collate_batch_data, setup_dataloader
from utils.scoring import bc_scoring, bc_score_names, nanmean

import torch as pt
import os
import numpy as np
from tqdm import tqdm
import wandb
from torchvision.ops import sigmoid_focal_loss


wandb.login(key='XXX')

class GeoFocalLoss(pt.nn.Module): 
    def __init__(self, alpha, beta, gamma, learn_beta=False):
        super(GeoFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.learn_beta = learn_beta

        beta_tensor = pt.tensor(beta, dtype=pt.float32)
        if learn_beta:
            self._beta_parameter = pt.nn.Parameter(beta_tensor)
        else:
            self.register_buffer('_fixed_beta', beta_tensor)

    @property
    def beta(self):
        if self.learn_beta:
            return 4.0 * pt.tanh(self._beta_parameter)
        else:
            return self._fixed_beta

    def forward(self, inputs, targets, dists):

        focal_loss = sigmoid_focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction='none')
        safe_dists = dists.clamp(min=1e-6)
        geo_term = pt.exp(-pt.pow(safe_dists, self.beta))
        geo_term[dists == 0] = 1e-6
        geo_loss = focal_loss * geo_term
        return geo_loss.mean()

def scoring(eval_results, device=pt.device('cpu')):
    # compute sum losses and scores for each entry
    losses, scores = [], []
    for loss, y, p in eval_results:
        losses.append(loss)
        scores.append(bc_scoring(y, p))

    # average scores
    m_losses = np.mean(losses)
    m_scores = nanmean(pt.stack(scores, dim=0)).numpy()

    # pack scores
    scores = {'loss': float(m_losses)}
    for i,s in enumerate(m_scores.squeeze(1)):
        scores[f'{bc_score_names[i]}'] = s
    return scores
    
def eval_step(model, device, batch_data, loss_fn, global_step):
    # unpack data
    onehot_seq, rmsf1, rmsf2, rsa, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, mapping, y, dists = [data.to(device) for data in batch_data]
    print (len(y))
    # run model
    z = model.forward(onehot_seq, rmsf1, rmsf2, rsa, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, mapping)
    # compute weighted loss
    loss = loss_fn(z, y, dists)

    return loss, y.detach(), pt.sigmoid(z).detach()


def train(config_data, config_model, config_runtime, output_path):
    wandb.init(project="residue-level codygat")
    
    # define device
    device = pt.device(config_runtime['device'])
    
    # create model
    model = Model(config_model)
    print(model)
    print(f"> {sum([int(pt.prod(pt.tensor(p.shape))) for p in model.parameters()])} parameters")
    
    model_filepath = '/home/omokhtari/public/ppi_model/model/model_xxx.pt'
    if os.path.isfile(model_filepath):
        model.load_state_dict(pt.load(model_filepath))
        global_step = 139775
    else:
        # starting global step
        global_step = 0
    
    # setup dataloaders - feature orders: onehot_seq, rmsf1, rmsf2, rsa, angular_variation, nn_topk, D_nn, R_nn, SCOD_nn, motion_v, motion_s, y
    dataloader_train = setup_dataloader(config_data, config_data['train_selection_filepath'])
    dataloader_test = setup_dataloader(config_data, config_data['valid_selection_filepath'])
    
    # send model to device
    model = model.to(device)
    
    # define loss fuction
    loss_fn = GeoFocalLoss(alpha=config_runtime['loss_alpha'], beta=config_runtime['loss_beta'], gamma=config_runtime['loss_gamma'], learn_beta=True)
    
    # define optimizer
    optimizer = pt.optim.Adam([*model.parameters(), *loss_fn.parameters()], lr=config_runtime["learning_rate"])
    
    # min loss initial value
    min_loss = 1e9
    patience_counter = 0
    
    # quick training step on largest data: memory check and pre-allocation
    batch_data = collate_batch_data([dataloader_train.dataset.get_largest()])
    optimizer.zero_grad()
    loss, _, _ = eval_step(model, device, batch_data, loss_fn, global_step)
    loss.backward()
    optimizer.step()
    
    # start training
    for epoch in range(config_runtime['num_epochs']):
        # train mode
        model = model.train()

        # train model
        train_results = []
        for batch_train_data in tqdm(dataloader_train):
            # global step
            global_step += 1
            
            # set gradient to zero
            optimizer.zero_grad()

            # forward & backward propagation
            loss, y, p = eval_step(model, device, batch_train_data, loss_fn, global_step)
            loss.backward()
            optimizer.step()

            # store evaluation results
            train_results.append([loss.detach().cpu(), y.cpu(), p.cpu()])
            
            if (global_step+1) % config_runtime["log_step"] == 1:
                # process evaluation results
                with pt.no_grad():
                    # scores evaluation results and reset buffer
                    scores = scoring(train_results, device=device)
                    scores_ = {f"{k}/train": v for k, v in scores.items()}
                    wandb.log(scores_, step=global_step)
                    train_results = []

                    # save model checkpoint
                    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
                    pt.save(model.state_dict(), model_filepath)

            # evaluation step
            if (global_step+1) % config_runtime["eval_step"] == 0:
                # evaluation mode
                model = model.eval()

                with pt.no_grad():
                    # evaluate model
                    test_results = []
                    for step_te, batch_test_data in enumerate(dataloader_test):
                        # forward propagation
                        losses, y, p = eval_step(model, device, batch_test_data, loss_fn, global_step)

                        # store evaluation results
                        test_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

                        # stop evaluating
                        if step_te >= config_runtime['eval_size']:
                            break

                    # scores evaluation results
                    scores = scoring(test_results, device=device)
                    scores_ = {f"{k}/valid": v for k, v in scores.items()}
                    wandb.log(scores_, step=global_step)
                    test_results = []

                    # save model and update min loss
                    current_loss = scores['loss']
                    if min_loss >= current_loss:
                        # update min loss
                        min_loss = current_loss
                        # save model
                        model_filepath = os.path.join(output_path, 'model_revision_MD.pt')
                        pt.save(model.state_dict(), model_filepath)
                        # Early stopping check
                        patience_counter = 0
                    else:
                        patience_counter += 1

                # back in train mode
                model = model.train()
        if patience_counter >= config_runtime['patience']:
            break  # Break out of the training loop
    wandb.finish()


train(config_data, config_model, config_runtime, '.')
