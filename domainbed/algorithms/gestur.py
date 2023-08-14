import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.optimizers import get_optimizer
from domainbed.networks.ur_networks import URFeaturizer
from domainbed.lib import misc
from domainbed.algorithms import Algorithm
from domainbed.algorithms.miro import ForwardModel


class Network(nn.Module):

    def __init__(self, featurizer, classifier):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.featurizer(x))

    def forward_feature(self, x):
        return self.featurizer(x)

    def classify_from_feature(self, x):
        return self.classifier(x)


class GESTUROpt(misc.PCGrad):

    def __init__(self, optimizer, lda):
        self._optim, self._lda = optimizer, lda
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        # grads[0]: grads from ce_loss
        # grads[1]: grads from reg_loss
        shared = torch.stack(has_grads).prod(0).bool()
        
        # sync scale between g_in and g_out
        g_in_norm = grads[0].norm()
        g_out_norm = grads[1].norm()
        if g_out_norm != 0:  # norm can be zero for the first step
            grads[1] = self._lda * g_in_norm * grads[1] / g_out_norm
        g_in, g_out = copy.deepcopy(grads)
        
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = torch.stack([g_in[shared], g_out[shared]]).mean(dim=0)
        merged_grad[~shared] = torch.stack([g_in[~shared], g_out[~shared]]).sum(dim=0)

        return merged_grad



class GESTUR(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = Network(self.featurizer, self.classifier)

        self.momentum_network = copy.deepcopy(self.network)

        self.checkpoint_ema = misc.CheckpointEMA(self.momentum_network, hparams.ema_decay)

        # optimizer
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.optimizer = GESTUROpt(self.optimizer, hparams.lda)

        self.return_ema = hparams.return_ema

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        logits = self.network(all_x)
        ce_loss = F.cross_entropy(logits, all_y)

        reg_loss = 0
        for params_s, params_t in zip(self.network.featurizer.parameters(), self.momentum_network.featurizer.parameters()):
            reg_loss += 0.5 * (params_s - params_t).norm(2) ** 2
        
        losses = [ce_loss, reg_loss]
        self.optimizer.zero_grad()
        self.optimizer.pc_backward(losses)
        self.optimizer.step()

        self.checkpoint_ema.step(self.network, self.momentum_network)
        return {"ce_loss": ce_loss.item(), "reg_loss": reg_loss.item()}

    def predict(self, x):
        if self.return_ema:
            return self.momentum_network(x)
        else:
            return self.network(x)

    def get_forward_model(self):
        if self.return_ema:
            forward_model = ForwardModel(self.momentum_network)
        else:
            forward_model = ForwardModel(self.network)
        return forward_model
