import random
import torch
import torch.nn.functional as F

class BinaryCrossEntropyLoss():
    """Binary Cross Entropy Loss
    """

    __negative_selection_rules__ = ("none","ohem","random","mix",)
    def __init__(self, negative_selection_rule:str='none', **kwargs):
        self.neg_select_rule = negative_selection_rule
        assert self.neg_select_rule in self.__negative_selection_rules__,"given negative selection rule is not defined"

        self.min_num_of_neg_ratio = kwargs.get('min_num_of_neg_ratio', 0.01)
        if self.neg_select_rule in ("ohem","mix"):
            # select 5 negatives for 1 positive using ohem selection
            self.ohem_ratio = kwargs.get('ohem_ratio', 5)
        if self.neg_select_rule in ("random","mix"):
            # select 5 negatives for 1 positive using random selection
            self.random_ratio = kwargs.get('random_ratio', 5)
        if self.neg_select_rule in ("random","ohem","mix"):
            # select 10 negatives for 1 positive
            self.neg_select_ratio = kwargs.get('neg_select_ratio', 10)

    def __call__(self, input:torch.Tensor, target:torch.Tensor):
        # input: torch.Tensor(N,)
        # target: torch.Tensor(N,)
        loss = F.binary_cross_entropy_with_logits(input,target, reduction='none')

        if self.neg_select_rule == 'none':
            return loss.mean()

        s_pos, = torch.where(target==1)
        s_neg, = torch.where(target==0)

        num_of_positives = s_pos.size(0)
        num_of_negatives = s_neg.size(0)
        min_num_of_negatives = int(target.size(0)*self.min_num_of_neg_ratio)
        max_num_of_negatives = num_of_negatives

        num_of_negatives = int(num_of_positives*self.neg_select_ratio)

        if self.neg_select_rule == 'random':
            # select negatives randomly
            num_of_negatives = int(num_of_positives*self.random_ratio)
            num_of_negatives = max(min_num_of_negatives, num_of_negatives)
            num_of_negatives = min(max_num_of_negatives, num_of_negatives)
            num_of_negatives = max(num_of_negatives,1) # for safety
            s_neg = random.sample(s_neg.cpu().numpy().tolist(), k=num_of_negatives)
            assert len(s_neg) > 0,"selected negative samples is zero, this may cause loss to become `nan`"
            if s_pos.size(0) == 0:
                return loss[s_neg].mean()
            return torch.cat([loss[s_neg],loss[s_pos]]).mean()
        elif self.neg_select_rule == 'ohem':
            # select negatives using ohem
            num_of_negatives = min(num_of_negatives, num_of_positives*self.ohem_ratio)
            num_of_negatives = max(min_num_of_negatives, num_of_negatives)
            num_of_negatives = min(max_num_of_negatives, num_of_negatives)
            num_of_negatives = max(num_of_negatives,1) # for safety
            assert num_of_negatives > 0,"selected negative samples is zero, this may cause loss to become `nan`"
            pos_loss = loss[s_pos]
            neg_loss = loss[s_neg]
            sorted_negs = neg_loss.argsort(descending=True)
            neg_loss = neg_loss[sorted_negs][:num_of_negatives]
            if pos_loss.size(0) == 0:
                return neg_loss.mean()
            return torch.cat([pos_loss,neg_loss]).mean()
        elif self.neg_select_rule == 'mix':
            # use ohem and random selection
            pos_loss = loss[s_pos]
            neg_loss = loss[s_neg]
            # ohem selection
            num_of_negatives = min(num_of_negatives//2, (num_of_positives//2)*self.ohem_ratio)
            num_of_negatives = max(min_num_of_negatives//2, num_of_negatives)
            num_of_negatives = min(max_num_of_negatives//2, num_of_negatives)
            num_of_negatives = max(num_of_negatives,1) # for safety
            sorted_negs = neg_loss.argsort(descending=True)
            ohem_loss = neg_loss[sorted_negs[:num_of_negatives]]
            #
            random_loss = neg_loss[sorted_negs[num_of_negatives:]]
            # random selection
            num_of_negatives = int(num_of_positives//2*self.random_ratio)
            num_of_negatives = max(min_num_of_negatives//2, num_of_negatives)
            num_of_negatives = min(random_loss.size(0), num_of_negatives)
            num_of_negatives = max(num_of_negatives,1) # for safety
            random_selection = random.sample(
                [i for i in range(random_loss.size(0))], k=num_of_negatives)

            random_loss = random_loss[random_selection]
            if pos_loss.size(0) == 0:
                return torch.cat([ohem_loss,random_loss], dim=0).mean()
            return torch.cat([ohem_loss,random_loss,pos_loss], dim=0).mean()
        else:
            raise AssertionError("this line should not be show up!!")