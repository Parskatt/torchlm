import torch
import time

def block_inverse(D,bs=3):
    #assume uniform blocksize
    m,_ = D.shape
    blocks = torch.stack([D[o:o+bs,o:o+bs] for o in range(0,m,bs)])
    block_inv = torch.linalg.inv(blocks)
    return torch.block_diag(*block_inv)

def solve_schur_ba(M, g, num_views, camera_param=12):
    k = camera_param*num_views
    gc,gp = g[:k],g[k:]
    A = M[:k,:k]
    B = M[k:,:k]
    assert torch.all(B == M[:k,k:].mT), "M is not symmetric"
    D = M[k:,k:] # 3x3 should be block diagonal
    D_inv = block_inverse(D)
    S = A - B.mT@D_inv@B

    hc = torch.linalg.solve(S,-gc+B.mT@D_inv@gp)

    hp = -D_inv@(gp+B@hc)
    return torch.cat((hc,hp))


if __name__ == "__main__":
    num_views = 13
    camera_param = 12
    num_3d_pt = 5000
    k = num_views*camera_param
    A = torch.randn(k,k).cuda()
    B = torch.randn(3*num_3d_pt,k).cuda()
    D = torch.block_diag(*torch.randn(3,3)[None].expand(num_3d_pt,3,3)).cuda()
    M = torch.zeros(k+3*num_3d_pt,k+3*num_3d_pt).cuda()
    M[:k,:k] = A
    M[:k,k:] = B.mT
    M[k:,:k] = B
    M[k:,k:] = D
    print(M.shape)
    g = torch.randn(k+3*num_3d_pt).cuda()
    t0=time.perf_counter()
    solve_schur_ba(M, g, num_views, camera_param)
    t1=time.perf_counter()
    #torch.linalg.solve(M,-g)
    t2 = time.perf_counter()
    print(t1-t0,t2-t1)
