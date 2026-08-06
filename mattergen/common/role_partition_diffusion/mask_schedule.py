from __future__ import annotations
import math
import torch

class AbsorbingMaskSchedule:
    """Cosine bar-alpha with exact all-MASK terminal state."""
    def __init__(self, steps: int = 32, kind: str = "cosine"):
        if kind != "cosine" or steps < 2: raise ValueError("only cosine schedules with steps>=2 are supported")
        self.steps = steps
        x=torch.arange(steps+1,dtype=torch.float64)/steps
        bar=torch.cos(x*math.pi/2).square();bar[0]=1.;bar[-1]=0.
        self.bar_alpha=bar.float()
    def sample_timestep(self, generator=None, terminal_probability=.25, device=None):
        if torch.rand((),generator=generator,device=device)<terminal_probability:return self.steps
        return int(torch.randint(1,self.steps+1,(),generator=generator,device=device))
    def rho(self,t):
        if t<1 or t>self.steps:raise ValueError('invalid reverse step')
        if t==1:return 1.
        return float((self.bar_alpha[t-1]-self.bar_alpha[t])/(1-self.bar_alpha[t]))
