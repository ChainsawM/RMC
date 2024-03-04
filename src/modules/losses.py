import torch
from torch import nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, base_margin=0.05):
        super().__init__()
        self.base_margin = base_margin

    def forward(self, embeddings, class_labels, irrelevance_levels, positive_irrelevance_levels=[1],
                similarity_type="cos"):

        sim_mat = self.pairwise_similarity(embeddings, similarity_type)
        mask = self.get_mask(class_labels, irrelevance_levels, positive_irrelevance_levels=positive_irrelevance_levels)
        margin = self.get_margin(class_labels, irrelevance_levels)

        sim_difference = sim_mat.unsqueeze(1) - sim_mat.unsqueeze(2) + margin
        sim_difference = sim_difference * mask
        loss = torch.clamp(sim_difference, min=0).sum()

        loss = loss / irrelevance_levels.tolist().count(0)

        return loss

    def pairwise_similarity(self, embeddings, similarity="cos"):
        if similarity == "cos":
            embeddings = F.normalize(embeddings, p=2.0, dim=1)
            sim_mat = torch.matmul(embeddings, embeddings.transpose(0, 1))
        elif similarity == "euclidean":
            ## we define the euclidean similairty as the negative euclidean distance
            sim_mat = -torch.norm(embeddings.unsqueeze(1) - embeddings.unsqueeze(0), dim=2, p=2)
        else:
            print("similarity not supported!")
            assert False

        return sim_mat

    def get_margin(self, class_labels, irrelevance_levels):
        modified_irrelevance_levels = irrelevance_levels.unsqueeze(0).masked_fill(class_labels.unsqueeze(0) != class_labels.unsqueeze(1), 4)
        irrelevance_levels_difference = modified_irrelevance_levels - irrelevance_levels.unsqueeze(1)
        return irrelevance_levels_difference.unsqueeze(0) * self.base_margin

    def get_mask(self, class_labels, irrelevance_levels, positive_irrelevance_levels=[1]):
        ## here the mask can be viewed as transparency
        # shape of mask: batch_size x batch_szie x batch_size (got by .unsqueeze(1).unsqueeze(2)), values are all one
        mask = torch.ones_like(class_labels).unsqueeze(1).unsqueeze(2).repeat(1, class_labels.size(0), class_labels.size(0))
        ## mask the non anchor points's triplet matrix
        mask = mask.masked_fill((irrelevance_levels != 0).unsqueeze(1).unsqueeze(2), 0.0)
        ## mask invalid triplet pairs for each valid anchor 
        ## mask invalid positive 
        mask = mask.masked_fill((class_labels.unsqueeze(0) != class_labels.unsqueeze(1)).unsqueeze(2), 0.0)

        temp_mask = torch.ones_like(irrelevance_levels) == 1
        for level in positive_irrelevance_levels:
            temp_mask = temp_mask & (irrelevance_levels != level)
        mask = mask.masked_fill(temp_mask.unsqueeze(0).unsqueeze(2), 0.0)

        ## mask invalid negative
        ## anchor point can never be negative
        mask = mask.masked_fill((irrelevance_levels == 0).unsqueeze(0).unsqueeze(1), 0.0)
        ## positive point of the same class as the positive can never be negative
        mask = mask.masked_fill(((class_labels.unsqueeze(0) == class_labels.unsqueeze(1)) & \
                                 (irrelevance_levels.unsqueeze(0) - irrelevance_levels.unsqueeze(1) <= 0)).unsqueeze(0),
                                0.0)

        # mask digonal elements 
        pos = torch.arange(class_labels.size(0)).to(class_labels.device)
        mask = mask.masked_fill((pos.unsqueeze(0) == pos.unsqueeze(1)).unsqueeze(2), 0.0)
        mask = mask.masked_fill((pos.unsqueeze(0) == pos.unsqueeze(1)).unsqueeze(1), 0.0)
        mask = mask.masked_fill((pos.unsqueeze(0) == pos.unsqueeze(1)).unsqueeze(0), 0.0)

        return mask
