from __future__ import annotations
import itertools,torch

def completions(partial: torch.Tensor):
    """Enumerate permutations consistent with target-row -> anchor-column partial tokens (-1 mask)."""
    k=partial.numel()
    if k>6:raise NotImplementedError('exact structured posterior supports K<=6 only')
    answer=[]
    for perm in itertools.permutations(range(k)):
        p=torch.tensor(perm,device=partial.device)
        if bool(((partial<0)|(partial==p)).all()):answer.append(p)
    if not answer:raise ValueError('partial matching has no legal completion')
    return torch.stack(answer)

def structured_nll(scores:torch.Tensor,truth:torch.Tensor,partial:torch.Tensor,temperature=1.):
    perms=completions(partial);values=scores[torch.arange(scores.shape[0],device=scores.device)[None,:],perms].sum(1)/temperature
    return -(scores[torch.arange(len(truth),device=scores.device),truth].sum()/temperature-torch.logsumexp(values,0))

def sample_completion(scores,partial,generator=None,temperature=1.):
    perms=completions(partial);value=scores[torch.arange(scores.shape[0],device=scores.device)[None,:],perms].sum(1)/temperature
    return perms[torch.multinomial(torch.softmax(value,0),1,generator=generator).item()]
