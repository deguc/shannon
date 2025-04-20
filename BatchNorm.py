class BatchNorm:
    
    def __init__(self, dim, momentum=0.9):
        gamma = np.ones(dim)
        beta = np.zeros(dim)

        self.ps = [gamma, beta]
        self.gs = make_zeros(self.ps)

        self.cashe=(None,None)

    def forward(self, x, train_flg=True):

        eps=1e-5
        
        gamma,beta=self.ps

        m = x.mean(axis=0)
        var = x.var(axis=0)
        std = np.sqrt(var + eps)

        x_centered = x - m
        std_inv = 1 / std

        x_norm = x_centered * std_inv
        out = gamma * x_norm + beta

        self.cashe=(x_centered,std_inv)

        return out

    def backward(self, dout):

        gamma,beta=self.ps
        
        x_centered,std_inv=self.cashe
        

        N = dout.shape[0]
        dx_ = dout * gamma
        dvar = np.sum(dx_ * x_centered * (-0.5) * std_inv**3, axis=0)
        dmu = np.sum(dx_ * (-std_inv), axis=0) + dvar * np.mean(-2. * x_centered, axis=0)

        dx = dx_ * std_inv + dvar * 2 * x_centered / N + dmu / N

        self.gs[0][...] = np.sum(dout * (x_centered * std_inv), axis=0)
        self.gs[1][...] = np.sum(dout, axis=0)

        return dx
