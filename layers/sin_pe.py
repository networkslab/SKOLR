        self.sin_L = kwargs.get('sin_L', 10)
        eps = 2 ** torch.arange(self.sin_L) * np.pi
        self.register_buffer('eps', eps)




    def _sin_pe(self, pe):
        raw_pe = pe
        pe = pe.unsqueeze(-1) # E x D x 1
        eps = self.eps.view(1, 1, -1)
        sin_pe = torch.sin(eps * pe)
        cos_pe = torch.cos(eps * pe)
        pe = torch.stack([sin_pe, cos_pe], dim=-1)

        return torch.cat([raw_pe, pe.flatten(1)], dim=-1)

