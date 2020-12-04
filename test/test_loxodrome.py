import math

import torch

from ais.loxodrome import Loxodrome


def random_init(*, batch_size, sog_kn=None, cog_deg=None):
    ones = torch.ones(batch_size)
    sog = ones * (torch.rand_like(ones) * 20. if sog_kn is None else sog_kn)
    cog = ones * (torch.rand_like(ones) * 360. if cog_deg is None else cog_deg)

    lat = (torch.rand_like(cog) * 180. - 90.) * .9
    lon = (torch.rand_like(cog) * 360. - 180.) * .9

    t = torch.rand_like(cog)

    lox = Loxodrome(batch_size=batch_size)
    state = lox.init_state(lat_deg=lat, lon_deg=lon, sog_kn=sog, cog_deg=cog)
    pred = lox.measurement_model(state, t)

    assert torch.allclose(state, pred)

    return state, t


def test_loxodrome_NORTH():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    batch_size = 1000
    sog_kn = 3.
    lox = Loxodrome(batch_size=batch_size)

    state, t = random_init(batch_size=batch_size, sog_kn=sog_kn, cog_deg=0.)
    state2 = lox.motion_model(state, t)

    assert torch.allclose((state2[:, 0] - state[:, 0]) / math.pi * 180., sog_kn / 60. * t)
    assert torch.allclose(state2[:, 1], state[:, 1])

    assert torch.allclose(state[:, 2], state2[:, 2])
    assert torch.allclose(state2[:, 3],
                          state[:, 3] * (1. + t * state[:, 2] * torch.tan(state[:, 0])))

    new_sog_rad = torch.sqrt(state2[:, 2] ** 2 + (state2[:, 3] * torch.cos(state2[:, 0])) ** 2)
    assert torch.allclose(new_sog_rad * 60. / math.pi * 180., torch.tensor([sog_kn, ]))


def test_loxodrome_EAST():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    batch_size = 1000
    sog_kn = 3.
    lox = Loxodrome(batch_size=batch_size)

    state, t = random_init(batch_size=batch_size, sog_kn=sog_kn, cog_deg=90.)
    state2 = lox.motion_model(state, t)

    assert torch.allclose(state[:, 0], state2[:, 0])
    assert torch.allclose((state2[:, 1] - state[:, 1]) / math.pi * 180.,
                          sog_kn / 60. * t / torch.cos(state[:, 0]))

    assert torch.allclose(state[:, 2], state2[:, 2])
    assert torch.allclose(state[:, 3], state2[:, 3])


def test_loxodrome_jacobian():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    batch_size = 1000
    lox = Loxodrome(batch_size=batch_size)

    state, t = random_init(batch_size=batch_size)
    tj = lox.motion_model_jacobian(state, t) - torch.eye(4, 4).unsqueeze(0).repeat(batch_size, 1, 1)
    j = tj / t.unsqueeze(1).unsqueeze(2)

    lat = state[:, 0]
    vlat = state[:, 2]
    vlon = state[:, 3]

    sec_lat = 1. / torch.cos(lat)
    tan_lat = torch.tan(lat)

    ones = torch.ones(batch_size)

    torch.allclose(j[:, 0, 0], ones * 0.)
    torch.allclose(j[:, 0, 1], ones * 0.)
    torch.allclose(j[:, 0, 2], ones * 1.)
    torch.allclose(j[:, 0, 3], ones * 0.)

    torch.allclose(j[:, 1, 0], ones * 0.)
    torch.allclose(j[:, 1, 1], ones * 0.)
    torch.allclose(j[:, 1, 2], ones * 0.)
    torch.allclose(j[:, 1, 3], ones * 1.)

    torch.allclose(j[:, 2, 0], ones * 0.)
    torch.allclose(j[:, 2, 1], ones * 0.)
    torch.allclose(j[:, 2, 2], ones * 0.)
    torch.allclose(j[:, 2, 3], ones * 0.)

    torch.allclose(j[:, 3, 0], ones * vlat * vlon * sec_lat ** 2)
    torch.allclose(j[:, 3, 1], ones * 0.)
    torch.allclose(j[:, 3, 2], ones * vlon * tan_lat)
    torch.allclose(j[:, 3, 3], ones * vlat * tan_lat)


def test_lon_innovation():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    def _diff(a, b):
        lox = Loxodrome(batch_size=1)
        diff_rad = lox.innovation(torch.tensor([[0., a / 180. * math.pi, 0., 0.]]),
                                  torch.tensor([[0., b / 180. * math.pi, 0., 0.]]))[0, 1]
        return diff_rad / math.pi * 180.

    d = _diff(-179., 179.)
    assert torch.allclose(d, torch.tensor(2.))

    d = _diff(179., -179.)
    assert torch.allclose(d, torch.tensor(-2.))

    d = _diff(89., -92.)
    assert torch.allclose(d, torch.tensor(-179.))

    d = _diff(-92., 89.)
    assert torch.allclose(d, torch.tensor(179.))
