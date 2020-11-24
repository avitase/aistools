import math
import os
import pathlib

import matplotlib.pyplot as plt
import torch

from .helper.data import DataLoader
from ..ais.ukf import UKF
from ..ais.ukf import UKFCell


def test_error():
    torch.manual_seed(0)

    def _test(*, measurement, prediction, expected):
        error = UKFCell(batch_size=1).error(
            measurement=torch.tensor(measurement, dtype=torch.double),
            prediction=torch.tensor(prediction, dtype=torch.double))

        assert torch.allclose(error, torch.tensor(expected, dtype=torch.double))

    _test(measurement=[[10., 10., 5., 60.], ],
          prediction=[[30., -20., 7., 55.], ],
          expected=[[-20., 30., -2., 5.], ])

    _test(measurement=[[175., 20., 5., 350.], ],
          prediction=[[-175., -175., 7., 10.], ],
          expected=[[-10., -165., -2., -20.], ])

    dtype = torch.double
    for _ in range(1000):
        size = torch.randint(low=1, high=10, size=(2,)).tolist()

        lat_m = torch.rand(size, dtype=dtype) * 360. - 180.
        lon_m = torch.rand(size, dtype=dtype) * 360. - 180.
        sog_m = torch.rand(size, dtype=dtype) * 100.
        cog_m = torch.rand(size, dtype=dtype) * 360.
        measurement = torch.stack((lat_m, lon_m, sog_m, cog_m,), dim=1)

        lat_p = torch.rand(size, dtype=dtype) * 360. - 180.
        lon_p = torch.rand(size, dtype=dtype) * 360. - 180.
        sog_p = torch.rand(size, dtype=dtype) * 100.
        cog_p = torch.rand(size, dtype=dtype) * 360.
        prediction = torch.stack((lat_p, lon_p, sog_p, cog_p,), dim=1)

        error = UKFCell(batch_size=1).error(measurement=measurement, prediction=prediction)

        assert torch.all(error > -180.)
        assert torch.all(error < +180.)


def test_cov_getter():
    torch.manual_seed(0)

    def _process_noise_cov_from_weights(*, size, weights, t):
        m = torch.zeros(size, dtype=weights.dtype)
        m[:, 0, 0] = .5 * t ** 2 * weights[:, 0]
        m[:, 1, 1] = .5 * t ** 2 * weights[:, 2]
        m[:, 2, 0] = t * weights[:, 3]
        m[:, 3, 1] = t * weights[:, 7]

        return torch.matmul(m, m.transpose(1, 2))

    def _measurement_noise_cov_from_weights(*, size, weights):
        m = torch.zeros(size, dtype=weights.dtype)
        m[:, 0, 0] = weights[:, 0]
        m[:, 1, 0] = weights[:, 1]
        m[:, 1, 1] = weights[:, 2]
        m[:, 2, 0] = weights[:, 3]
        m[:, 2, 1] = weights[:, 4]
        m[:, 2, 2] = weights[:, 5]
        m[:, 3, 0] = weights[:, 6]
        m[:, 3, 1] = weights[:, 7]
        m[:, 3, 2] = weights[:, 8]
        m[:, 3, 3] = weights[:, 9]

        return torch.matmul(m, m.transpose(1, 2))

    dtype = torch.double
    for _ in range(100):
        batch_size = torch.randint(low=1, high=10, size=(1,)).item()
        pw = torch.rand(batch_size, 10, dtype=dtype)
        mw = torch.rand(batch_size, 10, dtype=dtype)
        t = torch.rand(batch_size, dtype=dtype)

        p_cov = _process_noise_cov_from_weights(size=(batch_size, 4, 4), weights=pw, t=t)
        m_cov = _measurement_noise_cov_from_weights(size=(batch_size, 4, 4), weights=mw)

        ukf = UKF(batch_size)
        ukf.process_noise = pw
        ukf.measurement_noise = mw
        assert torch.allclose(p_cov, ukf.process_noise_cov(t))
        assert torch.allclose(m_cov, ukf.measurement_noise_cov())


def test_ukf():
    torch.manual_seed(0)
    batch_size = 3

    img_dir = pathlib.Path('img')
    if not img_dir.is_dir():
        os.mkdir(img_dir)

    data_loader = DataLoader(batch_size=batch_size)

    for b in range(batch_size):
        lox = data_loader.lox_gt[b]
        ais = data_loader.ais_gt[b]

        fig, axs = plt.subplots(2, 4)
        fig.set_size_inches(12., 6.)
        fig.subplots_adjust(wspace=.4, hspace=.4)
        for i in range(4):
            axs[0, i].plot(lox[i], marker='.')
            axs[0, i].set_title(f'Lox. {i + 1}')

        for i in range(4):
            axs[1, i].plot(ais[i], marker='.')
            axs[1, i].set_title(f'AIS {i + 1}')

        fig.savefig(img_dir / f'components_b{b + 1}.png')

        fig, ax = plt.subplots()
        ax.plot(torch.cos(ais[0] / 180. * math.pi) * ais[1], ais[0], marker='.')
        ax.set_aspect(1., 'datalim')
        ax.set_xlabel('cos(Latitude) x Longitude')
        ax.set_ylabel('Latitude')
        fig.savefig(img_dir / f'trajectory_b{b + 1}.png')
