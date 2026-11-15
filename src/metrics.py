"""Separation metrics: scale-invariant SDR, SIR and SAR (Le Roux et al., 2019).

The references are rescaled to each estimate, and the error is split by projection onto all
references. The paper used bss_eval (filter length 512), so its SIR and SAR differ slightly
from these; SI-SDR is the same measure.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

EPS = 1e-8


def si_sdr(estimates, targets):
    """SI-SDR (dB) of every estimate against every target, ``(B, K, L), (B, M, L) -> (B, K, M)``."""
    e = estimates.unsqueeze(2)
    t = targets.unsqueeze(1)
    gain = (e * t).sum(-1, keepdim=True) / ((t * t).sum(-1, keepdim=True) + EPS)
    proj = gain * t
    return 10 * torch.log10((proj.pow(2).sum(-1) + EPS) / ((e - proj).pow(2).sum(-1) + EPS))


def match_sources(estimates, targets):
    """Pick and order the K >= M estimates that best match the M targets (max total SI-SDR).

    Returns the matched estimates ``(B, M, ...)`` and the chosen indices ``(B, M)``.
    """
    flat_e = estimates.flatten(2)
    flat_t = targets.flatten(2)
    score = si_sdr(flat_e, flat_t).detach().cpu().numpy()
    index = np.stack([linear_sum_assignment(-s.T)[1] for s in score])
    index = torch.from_numpy(index).to(estimates.device)
    gather = index.view(*index.shape, *([1] * (estimates.dim() - 2))).expand(
        -1, -1, *estimates.shape[2:])
    return torch.gather(estimates, 1, gather), index


def bss_measures(estimates, targets):
    """SI-SDR, SI-SIR and SI-SAR (dB), each ``(B, M)``, of matched estimates ``(B, M, L)``."""
    m = targets.shape[1]
    gram = targets @ targets.transpose(-1, -2) + EPS * torch.eye(m, device=targets.device)
    sdr, sir, sar = [], [], []
    for j in range(m):
        est = estimates[:, j]
        t = targets[:, j]
        gain = (est * t).sum(-1, keepdim=True) / ((t * t).sum(-1, keepdim=True) + EPS)
        target = gain * t
        # Projection of the estimate on all references; what is outside is artifacts.
        coef = torch.linalg.solve(gram, (targets * est.unsqueeze(1)).sum(-1))
        proj_all = (targets * coef.unsqueeze(-1)).sum(1)
        interf = proj_all - target
        artif = est - proj_all
        e_target, e_interf, e_artif = (v.pow(2).sum(-1) for v in (target, interf, artif))
        sdr.append(10 * torch.log10((e_target + EPS) / (e_interf + e_artif + EPS)))
        sir.append(10 * torch.log10((e_target + EPS) / (e_interf + EPS)))
        sar.append(10 * torch.log10((e_target + e_interf + EPS) / (e_artif + EPS)))
    return torch.stack(sdr, 1), torch.stack(sir, 1), torch.stack(sar, 1)
