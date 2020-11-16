import torch

from nav.ukf import UKF


def test_error():
    torch.manual_seed(0)

    def _test(*, measurement, prediction, expected):
        error = UKF(batch_size=1).error(measurement=torch.tensor(measurement, dtype=torch.double),
                                        prediction=torch.tensor(prediction, dtype=torch.double))

        assert torch.allclose(error, torch.tensor(expected, dtype=torch.double))

    _test(measurement=[[10., 10., 5., 60.], ],
          prediction=[[30., -20., 7., 55.], ],
          expected=[[-20., 30., -2., 5.], ])

    _test(measurement=[[175., 20., 5., 350.], ],
          prediction=[[-175., -175., 7., 10.], ],
          expected=[[-10., -165., -2., -20.], ])

    dtype = torch.double
    for _ in range(10000):
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

        error = UKF(batch_size=1).error(measurement=measurement, prediction=prediction)

        assert torch.all(error > -180.)
        assert torch.all(error < +180.)
