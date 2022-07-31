import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):

    def __init__(self, s=20.0, m=0.2, verbal=False, loss_weight=1.0):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.iter = 0
        self.verbal = verbal
        self.loss_weight = loss_weight

    def forward(self, input, labels):
        # input: size = B x num_class
        cos = input
        one_hot = torch.zeros(cos.size()).type_as(input)
        one_hot = one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = self.s * (cos - one_hot * self.m)

        softmax_output = F.log_softmax(output, dim=1)
        loss = -1 * softmax_output.gather(1, labels.view(-1, 1))
        loss = loss.mean()

        if self.iter % 10 == 0 and self.verbal:
            angles = cos.data.acos()
            angles_non_target = torch.sum((1 - one_hot) * angles, dim=1) / (angles.shape[1] - 1)
            angles_non_target_mean = angles_non_target.mean()
            angles_non_target_min = angles_non_target.min()
            angles_non_target_max = angles_non_target.max()

            angles_target = angles.gather(1, labels.view(-1, 1))
            angles_target_mean = angles_target.mean()
            angles_target_min = angles_target.min()
            angles_target_max = angles_target.max()
            print('angle_target:%f (min:%f, max:%f), angle_non_target:%f (min:%f, max:%f)' %
                  (angles_target_mean, angles_target_min, angles_target_max, angles_non_target_mean,
                   angles_non_target_min, angles_non_target_max))
        self.iter += 1
        return loss * self.loss_weight
