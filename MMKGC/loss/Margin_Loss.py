import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Margin_Loss(nn.Module):
    """Margin Ranking Loss

    Attributes:
        args: Some pre-set parameters, etc 
        model: The KG model for training.
    """
    def __init__(self, args, model):
        super(Margin_Loss, self).__init__()
        self.args = args
        self.model = model
        self.loss = nn.MarginRankingLoss(self.args.margin)

    def forward(self, pos_score, neg_score):
        """Creates a criterion that measures the loss given inputs pos_score and neg_score. In math:
        
        \text{loss}(x1, x2, y) = \max(0, -y * (x1 - x2) + \text{margin})
        
        Args:
            pos_score: The score of positive samples.
            neg_score: The score of negative samples.
        Returns:
            loss: The training loss for back propagation.
        """
        label = torch.Tensor([1]).type_as(pos_score)
        loss = self.loss(pos_score, neg_score, label)
            
        return loss

class MarginLoss(nn.Module):

	def __init__(self, args, model):
		super(MarginLoss, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([args.margin]))
		self.margin.requires_grad = False
		
	
	def get_weights(self, n_score):
		return F.softmax(-n_score * 2.0, dim = -1).detach()

	def forward(self, p_score, n_score):
		
		return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
		
	