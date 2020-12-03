import math

import torch

from ais.loxodrome import Loxodrome


def random_init(*, batch_size, sog_kn=None, cog_deg=None):
    ones = torch.ones(batch_size, 1)
    sog = ones * (torch.rand_like(ones) * 20. if sog_kn is None else sog_kn)
    cog = ones * (torch.rand_like(ones) * 360. if cog_deg is None else cog_deg)

    lat = torch.rand_like(cog) * 360. - 180.
    lon = torch.rand_like(cog) * 360. - 180.

    t = torch.rand_like(cog).squeeze(1)

    return Loxodrome(batch_size=batch_size).init_state(lat_deg=lat * .9,
                                                       lon_deg=lon,
                                                       sog_kn=sog,
                                                       cog_deg=cog).squeeze(2), t


def test_loxodrome_NORTH():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    batch_size = 1000
    sog_kn = 3.
    lox = Loxodrome(batch_size=batch_size)

    state, t = random_init(batch_size=batch_size, sog_kn=sog_kn, cog_deg=0.)
    state2 = lox.motion_model(state, t)

    assert torch.allclose(state2[:, 0], state[:, 0] + sog_kn / 60. * t)

    sign = torch.sign(state[:, 1] * torch.tan(state[:, 0] / 180 * math.pi))
    assert torch.all(sign * (state2[:, 1] - state[:, 1]) < 0.)

    assert torch.allclose(state[:, (2, 3)], state2[:, (2, 3)])

    assert torch.allclose(state[:, 2], torch.ones_like(state[:, 2]) * sog_kn)
    assert torch.allclose(state[:, 3], torch.zeros_like(state[:, 3]))


def test_loxodrome_EAST():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    batch_size = 1000
    sog_kn = 3.
    lox = Loxodrome(batch_size=batch_size)

    state, t = random_init(batch_size=batch_size, sog_kn=sog_kn, cog_deg=90.)
    state2 = lox.motion_model(state, t)

    assert torch.allclose(state[:, 0], state2[:, 0])
    assert torch.allclose(state2[:, 1], state[:, 1] + sog_kn / 60. * t)

    assert torch.allclose(state[:, (2, 3)], state2[:, (2, 3)])

    assert torch.allclose(state[:, 2], torch.zeros_like(state[:, 2]))
    assert torch.allclose(state[:, 3], torch.ones_like(state[:, 3]) * sog_kn)


def test_loxodrome_jacobian():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    batch_size = 1000
    lox = Loxodrome(batch_size=batch_size)

    state, t = random_init(batch_size=batch_size)
    tj = lox.motion_model_jacobian(state, t) - torch.eye(4, 4).unsqueeze(0).repeat(batch_size, 1, 1)
    j = tj / t.unsqueeze(1).unsqueeze(2)

    lat_deg = state[:, 0]
    lat_rad = lat_deg / 180. * math.pi

    slon_deg = state[:, 1]

    vlat_kn = state[:, 2]
    vlat_deg = vlat_kn / 60.

    ones = torch.ones(batch_size)

    torch.allclose(j[:, 0, 0], ones * 0.)
    torch.allclose(j[:, 0, 1], ones * 0.)
    torch.allclose(j[:, 0, 2], ones * 1.)
    torch.allclose(j[:, 0, 3], ones * 0.)

    torch.allclose(j[:, 1, 0], ones * -slon_deg * vlat_deg / torch.cos(lat_rad) ** 2)
    torch.allclose(j[:, 1, 1], ones * -vlat_deg * torch.tan(lat_rad))
    torch.allclose(j[:, 1, 2], ones * -slon_deg * torch.tan(lat_rad))
    torch.allclose(j[:, 1, 3], ones * 1.)

    torch.allclose(j[:, 2, 0], ones * 0.)
    torch.allclose(j[:, 2, 1], ones * 0.)
    torch.allclose(j[:, 2, 2], ones * 0.)
    torch.allclose(j[:, 2, 3], ones * 0.)

    torch.allclose(j[:, 3, 0], ones * 0.)
    torch.allclose(j[:, 3, 1], ones * 0.)
    torch.allclose(j[:, 3, 2], ones * 0.)
    torch.allclose(j[:, 3, 3], ones * 0.)


def test_measurement_model():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    batch_size = 1000
    lox = Loxodrome(batch_size=batch_size)

    state, t = random_init(batch_size=batch_size)
    state2 = lox.measurement_model(state, t)

    assert torch.allclose(state, state2)
