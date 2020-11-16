import math

import torch

from nav import loxodrome


def test_loxodrome():
    torch.manual_seed(0)

    def _random_init(sog):
        lat = torch.rand_like(sog, dtype=sog.dtype) * 360. - 180.
        lon = torch.rand_like(sog, dtype=sog.dtype) * 360. - 180.
        cog = torch.rand_like(sog, dtype=sog.dtype) * 360.
        return loxodrome.init_from_ais(lat_deg=lat, lon_deg=lon, sog_kn=sog, cog_deg=cog)

    def _deg2nm(deg):
        return deg * 60.

    def _rad2nm(rad):
        return _deg2nm(rad / math.pi * 180.)

    def _lox_dist_nm(lox1, lox2):
        lat1 = lox1[:, 0]
        lat2 = lox2[:, 0]
        vlat1 = lox1[:, 2]
        vlat2 = lox2[:, 2]
        scaled_vlon1 = lox1[:, 3]
        scaled_vlon2 = lox2[:, 3]

        assert torch.allclose(vlat1, vlat2)
        assert torch.allclose(scaled_vlon1, scaled_vlon2)

        cog = torch.atan2(scaled_vlon1, vlat1)
        return torch.abs(_rad2nm(lat2 - lat1) / torch.cos(cog))

    dtype = torch.double
    for _ in range(10000):
        size = torch.randint(low=1, high=10, size=(3,)).tolist()
        sog = torch.rand(size, dtype=dtype) * 100.
        t = torch.rand(size, dtype=dtype) * 3600 * 10.

        lox = _random_init(sog)
        lox2 = loxodrome.advance(lox, t=t)
        s = _lox_dist_nm(lox, lox2)
        assert torch.allclose(s, t * sog)
