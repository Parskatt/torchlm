import torch
from functorch import jacrev, jacfwd

#TODO: stuff
def lm(residual, theta_0, *args, jac_mode="forward", lam_0 = 10, num_iter = 100, lam_min = 1e-7, lam_max = 1e7, L = 2):
    jac = jacfwd(residual) if jac_mode == "forward" else jacrev(residual)
    theta = theta_0.clone()
    lam = lam_0
    r = residual(theta,*args)
    loss = (r**2).mean()
    for i in range(num_iter):
        print(lam,loss.item())
        loss_last = loss
        J = jac(theta,*args)
        JTJ = J.mT@J
        damp = lam*(torch.eye(100)+torch.diag(torch.diag(JTJ)))
        step = torch.linalg.solve(JTJ+damp,-J.mT@r)
        theta = theta + step
        r = residual(theta,*args)
        loss = (r**2).mean()
        if loss > loss_last:
            loss = loss_last
            theta = theta - step # go back!
            lam = min(lam_max, lam*L)
        else:
            lam = max(lam_min, lam/L)
    return theta


def test_residual(theta, target):
    return theta-target

def test_residual2(theta, target):
    return (theta**2+theta)-target


theta = torch.randn(100)
target = torch.randn(100)
weirdos = torch.randn(100)
#lm(test_residual, theta, target)
lm(test_residual2, theta, target)
