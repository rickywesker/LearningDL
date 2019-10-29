#SV-AM-Softmax
class SVAMLinear(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 num_class,
                 t = 1.2,
                 m = 0.35,
                 scale = 30):
        super(SVLinear,self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.t = t
        self.m = m
        self.scale = scale
        self.weights = Parameter(torch.Tensor(in_channels, num_class))
        self.weights.data.uniform_(-1,1).renorm(2,1,1e-5).mul(1e5)
    def forward(self,input,target):
        norm_weights = F.normalize(self.weights,dim=0)
        cos_theta = torch.mm(input,norm_weights)#test
        batch_size = target.size(0)
        gtScore = cos_theta[torch.arange(0,batch_size),target].view(-1,1)
        mask = cos_theta > (gtScore - self.m)
        finalScore = torch.where(gtScore > self.m, gtScore - self.m,gtScore)
        hardEx = cos_theta[mask]
        cos_theta[mask] = self.t * hardEx + self.t - 1.0
        cos_theta.scatter_(1, target.data.view(-1,1),finalScore)
        cos_theta *= self.scale
        return cos_theta