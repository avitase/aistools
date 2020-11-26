import warnings

import torch

from ...ais import loxodrome


class DataLoader:
    def __init__(self, *, batch_size, n, debug=False):
        lat0_deg_gt = (torch.rand(batch_size) * 2. - 1.) * (70. if debug else 90.)
        lon0_deg_gt = (torch.rand(batch_size) * 2. - 1.) * (160. if debug else 180.)
        sog0_kn_gt = torch.rand(batch_size) * 20. + 5.
        cog0_deg_gt = torch.rand(batch_size) * (300. if debug else 360.)

        loxs_gt = torch.empty(batch_size, 4, n)
        loxs_gt[:, :, 0] = loxodrome.init_from_ais(lat_deg=lat0_deg_gt,
                                                   lon_deg=lon0_deg_gt,
                                                   sog_kn=sog0_kn_gt,
                                                   cog_deg=cog0_deg_gt)

        if debug:
            warnings.warn('Perturbation disabled')

        dsog = torch.zeros(batch_size, n) if debug else .5 * torch.normal(
            torch.zeros(batch_size, n),
            torch.ones(batch_size, n))
        dcog = torch.zeros(batch_size, n) if debug else torch.normal(torch.zeros(batch_size, n),
                                                                     torch.ones(batch_size, n)) * 1.
        dt = torch.ones(batch_size, n)

        for i in range(1, n):
            ais = loxodrome.to_ais(loxs_gt[:, :, i - 1])
            ais[:, 2] += dsog[:, i] * dt[:, i]
            ais[:, 3] += dcog[:, i] * dt[:, i]
            ais[:, 3] = torch.fmod(ais[:, 3], 360.)
            lox = loxodrome.init_from_ais(lat_deg=ais[:, 0],
                                          lon_deg=ais[:, 1],
                                          sog_kn=ais[:, 2],
                                          cog_deg=ais[:, 3])
            loxs_gt[:, :, i] = loxodrome.advance(lox.unsqueeze(2),
                                                 t=dt[:, i].unsqueeze(1)).squeeze(2)

        self.dt = dt
        self.lox_gt = loxs_gt
        self.ais_gt = loxodrome.to_ais(loxs_gt)
