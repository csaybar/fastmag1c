"""Matched filters for a batch of CCD columns.

Every function takes x with shape (B, N, s): B columns, N pixels per column
and s bands, and a unit CH4 absorption template t with shape (s,). They
return the enhancement (B, N, 1) in ppm·m and the albedo factor (B, N, 1).

w is an optional (B, N, 1) mask with 1 for valid pixels and 0 for pixels
left out of the column statistics. Their output is meaningless, and x must
still be finite there (the column mean is a good fill).
"""

import torch

EPS = 1e-9


def _mean(x, w):
    if w is None:
        return x.mean(dim=1, keepdim=True)
    return (x * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True)


def _regularize(C, alpha):
    # (1 - alpha) C + alpha diag(C)
    return C.lerp_(torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1)), alpha)


def _covariance(U, n, alpha):
    C = torch.bmm(U.transpose(1, 2), U) / n
    return _regularize(C, alpha)


def _solve(C, tau):
    L = torch.linalg.cholesky(C)
    return torch.cholesky_solve(tau.transpose(1, 2), L)


def _count(x, w):
    return x.shape[1] if w is None else w.sum(dim=1, keepdim=True)


def _masked(U, w):
    return U if w is None else U * w


@torch.no_grad()
def rmf(x, t, alpha=1e-4, scaling=1e5, w=None):
    """Albedo-corrected matched filter, single pass."""
    n = _count(x, w)
    mu = _mean(x, w)
    tau = t * mu
    Ci_tau = _solve(_covariance(_masked(x - mu, w), n, alpha), tau)
    norm = torch.bmm(tau, Ci_tau).clamp_(min=EPS)
    R = torch.bmm(x, mu.transpose(1, 2)) / torch.bmm(mu, mu.transpose(1, 2))
    mf = torch.bmm(x - mu, Ci_tau) / (R * norm)
    return mf.relu_() * scaling, R


@torch.no_grad()
def mag1c(x, t, num_iter=30, alpha=1e-4, gamma=1.0, scaling=1e5, w=None):
    """Reference MAG1C. Recomputes the covariance from scratch every iteration."""
    n = _count(x, w)
    mf, R = rmf(x, t, alpha, scaling=1.0, w=w)
    tau = t * _mean(x, w)

    for _ in range(num_iter):
        Z = x - gamma * R * mf * tau
        mu = _mean(Z, w)
        tau = t * mu

        Ci_tau = _solve(_covariance(_masked(Z - mu, w), n, alpha), tau)
        norm = torch.bmm(tau, Ci_tau).clamp_(min=EPS)
        reg = 1.0 / (R * (mf + EPS))
        mf = (torch.bmm(x - mu, Ci_tau) - reg) / (R * norm)
        mf.relu_()

    return mf * scaling, R


@torch.no_grad()
def fastmag1c(x, t, num_iter=30, alpha=1e-4, gamma=1.0, scaling=1e5, w=None):
    """MAG1C with the rank-2 covariance update. Same output as mag1c()."""
    n = _count(x, w)

    # base covariance, O(N s^2), computed once
    mu0 = _mean(x, w)
    U = x - mu0
    Uw = _masked(U, w)
    C0 = torch.bmm(Uw.transpose(1, 2), Uw) / n

    mf, R = rmf(x, t, alpha, scaling=1.0, w=w)
    tau = t * mu0

    for _ in range(num_iter):
        d = gamma * R * mf
        d_bar = _mean(d, w)
        delta = _masked(d - d_bar, w)

        # rank-2 correction, O(N s)
        a = (torch.bmm(delta.transpose(1, 2), Uw) / n).squeeze(1)
        beta = ((delta * delta).sum(dim=1, keepdim=True) / n).squeeze(-1)
        tp = tau.squeeze(1)
        P = torch.stack([a, tp], dim=2)
        Q = torch.stack([-tp, beta * tp - a], dim=2)
        C = _regularize(C0 + torch.bmm(P, Q.transpose(1, 2)), alpha)

        shift = d_bar * tau
        mu = mu0 - shift
        tau = t * mu

        Ci_tau = _solve(C, tau)
        norm = torch.bmm(tau, Ci_tau).clamp_(min=EPS)
        reg = 1.0 / (R * (mf + EPS))
        mf = (torch.bmm(U + shift, Ci_tau) - reg) / (R * norm)
        mf.relu_()

    return mf * scaling, R
