import numpy as np
import torch


def orthonormal_basis(a, b, c, norm=True):
    """ Find an orthonormal basis as a matrix of [a,b,c].
    Useful for constructing rotation matrices between planes according to the first answer here:
    https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b

    :param a: (batch, 3) or (3, ). point(s) of the plane
    :param b: (batch, 3) or (3, ). point(s) of the plane
    :param c: (batch, 3) or (3, ). point(s) of the plane

    :return: orthonormal basis as a matrix of [e1, e2, e3]. calculated as:
    """
    v1_ = c - b
    v2_ = b - a
    v3_ = torch.cross(v1_, v2_, dim=-1)
    v2_ready = torch.cross(v3_, v1_, dim=-1)
    basis = torch.stack([v1_, v2_ready, v3_], dim=-2)
    if norm:
        basis /= torch.norm(basis, dim=-1, keepdim=True)
    return basis


def mp_nerf_torch(a, b, c, l, theta, chi):
    """ Natural extension of Reference Frame.

        :param a: (batch, 3) or (3,). point(s) of the plane, not connected to d
        :param b: (batch, 3) or (3,). point(s) of the plane, not connected to d
        :param c: (batch, 3) or (3,). point(s) of the plane, connected to d
        :param l: ??
        :param torch.Tensor theta: (batch,) or (float).  angle(s) between b-c-d
        :param chi: (batch,) or float. dihedral angle(s) between the a-b-c and b-c-d planes
        :return: d (batch, 3) or (float). the next point in the sequence, linked to c
    """

    if not ((-np.pi <= theta) * (theta < np.pi)).all():
        raise ValueError(f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}")
    ba = b - a
    cb = c - b

    n_plane = torch.cross(ba, cb, dim=-1)
    rotate = torch.stack([cb, torch.cross(n_plane, cb, dim=-1), n_plane], dim=-1)

    rotate /= torch.norm(rotate, dim=-2, keepdim=True)

    d = torch.stack([-torch.cos(theta),
                     torch.sin(theta) * torch.cos(chi),
                     torch.sin(theta) * torch.sin(chi)], dim=-1).unsqueeze(-1)

    return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()
