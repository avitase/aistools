import torch

from ...ais import loxodrome


class DataLoader:
    def __init__(self, *, batch_size):
        lat0_deg_gt = (torch.rand(batch_size) * 2. - 1.) * 70.
        lon0_deg_gt = torch.rand(batch_size) * 360.
        sog0_kn_gt = torch.rand(batch_size) * 20. + 5.
        cog0_deg_gt = torch.rand(batch_size) * 360.

        n = 45
        loxs_gt = torch.empty(batch_size, 4, n)
        loxs_gt[:, :, 0] = loxodrome.init_from_ais(lat_deg=lat0_deg_gt,
                                                   lon_deg=lon0_deg_gt,
                                                   sog_kn=sog0_kn_gt,
                                                   cog_deg=cog0_deg_gt)

        dsog = torch.normal(torch.zeros(batch_size, n), torch.ones(batch_size, n)) * .5
        dcog = torch.normal(torch.zeros(batch_size, n), torch.ones(batch_size, n)) * 1.
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
            loxs_gt[:, :, i] = loxodrome.advance(lox, t=dt[:, i])

        self.lox_gt = loxs_gt
        self.ais_gt = loxodrome.to_ais(loxs_gt)
