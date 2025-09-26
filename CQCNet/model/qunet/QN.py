import torch
from torch.nn import Module, Parameter


class QuaternionNorm2d(Module):

    def __init__(self, num_features, num_groups=4, gamma_init=1., beta_param=True, momentum=0.1):
        super(QuaternionNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.eps = torch.tensor(1e-5)
        self.num_groups = num_groups
        self.momentum = momentum

    def forward(self, input):
        G = self.num_groups

        b_s, ch, ha, wa = input.size()
        input = input.reshape(b_s, 4, int(ch / 4), ha, wa)

        x = input.reshape(b_s, G, -1)
        mean = x.mean(-1, keepdim=True)

        delta = x - mean  # b 4
        quat_variance = (delta ** 2).mean(dim=2).mean(dim=1)
        quat_variance = quat_variance.unsqueeze(1).unsqueeze(1)
        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        normalized = delta / denominator
        normalized = normalized.view(b_s, ch, ha, wa)

        gamma = self.gamma.repeat(1, 4, 1, 1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        output = (gamma * normalized) + self.beta

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma.shape) \
               + ', beta=' + str(self.beta.shape) \
               + ', eps=' + str(self.eps.shape) + ')'
