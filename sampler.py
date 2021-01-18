import torch
import torch.nn as nn
import numpy as np


class AdversarySampler:
    def __init__(self, budget, args):
        self.budget = budget
        self.args = args

    def OUI(self, v):
        v = nn.functional.softmax(v, dim=1)
        v_max, idx = torch.max(v, 1)
        c = v.size()[1]
        min_var = ((v_max-1/c)**2+(c-1)*(1/c-(1-v_max)/(c-1))**2)/c
        var_v = torch.var(v, 1)
        indicator = 1 - min_var / var_v * v_max
        indicator = indicator.mean(1).mean(1)
        return indicator.detach()

    def sample(self, task_model, data, cuda):
        all_preds = []
        all_indices = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for images, _, indices in data:
            images = images.to(device)

            with torch.no_grad():
                preds = task_model(images)[0]
                preds = self.OUI(preds)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        # all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices
