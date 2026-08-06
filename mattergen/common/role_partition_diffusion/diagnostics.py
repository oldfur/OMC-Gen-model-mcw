from __future__ import annotations
import torch
def matrix_residuals(R,C,M):
    R=R.float();one=torch.ones((R.shape[0],R.shape[1]),device=R.device,dtype=R.dtype)
    return {'C_symmetric':float((C-C.T).abs().max()),'C_diagonal':float((C.diag()-1).abs().max()),'C_row_sum':float((C.sum(1)-M).abs().max()),'C_idempotence':float((C@C-M*C).abs().max()),'CR':float((C@R-one).abs().max())}
