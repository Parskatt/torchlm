from typing import Any, Callable
import torch
from torch import Tensor
from functorch import jacrev, jacfwd

#TODO: stuff
# See http://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf and https://people.duke.edu/~hpgavin/ce281/lm.pdf for some further reading
def lm(residual: Callable, theta_0: Tensor, *args: Any, jac_mode : str = "forward", lam_0 : float = 10., 
       num_iter: int = 100, lam_min: float = 1e-7, lam_max: float = 1e7, L: float = 2., verbose = False,
       ftol = 1e-6, lam_momentum: float = 0.):
    jac = jacfwd(residual) if jac_mode == "forward" else jacrev(residual)
    theta = theta_0.clone()
    lam = lam_0
    r = residual(theta,*args)
    loss = (r**2).mean()
    for i in range(num_iter):
        if verbose:
            print(f"At iter {i}: {lam=}, loss = {loss.item()}")
        # If jacobian is sparse, this is quite inefficient.
        J = jac(theta,*args)
        n,m = J.shape
        JTJ = J.mT@J
        damp = lam*(torch.eye(m)+torch.diag(torch.diag(JTJ)))
        step = torch.linalg.solve(JTJ+damp,-J.mT@r)
        theta = theta + step
        new_r = residual(theta,*args)
        new_loss = (new_r**2).mean()
        if (loss-new_loss).abs() < ftol:
            if verbose:
                print("ftol, stopping")
            break
        if new_loss > loss:
            if lam == lam_max:
                # madness is doing the same thing over and over,
                # and expecting different results
                if verbose:
                    print("Reached maximum lambda, stopping")
                break
            theta = theta - step # go back!
            lam = min(lam_max, lam_momentum*lam + (1-lam_momentum)*lam*L)
        else:
            lam = max(lam_min, lam_momentum*lam + (1-lam_momentum)*lam/L)
            loss, r = new_loss, new_r
    return theta


def test_residual(theta, target):
    return theta-target

def test_residual2(theta, target, A):
    return A@(theta)-target

def test_residual3(theta, target, A):
    return A@(theta**2+theta)-target

if __name__ == "__main__":
    theta = torch.randn(1000)
    target = torch.randn(10)
    A = torch.randn(10,1000)
    #lm(test_residual, theta, target)
    #lm(test_residual3, theta, target, A, verbose=True, L=1, lam_0=10)
    import time
    t0 = time.perf_counter()
    lm(test_residual3, theta, target, A, verbose=True, jac_mode="forward")
    t1 = time.perf_counter()
    print(t1-t0)
    #lm(test_residual3, theta, target, A, verbose=True, lam_momentum=0.5)
