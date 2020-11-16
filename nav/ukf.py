import torch

from nav import loxodrome
from ukf.ukf import UKFCell, KFRNN


class UKF(UKFCell):
    def __init__(self, batch_size: int):
        super(UKF, self).__init__(batch_size=batch_size,
                                  state_size=4,
                                  measurement_size=4,
                                  log_cholesky=True)

    def motion_model(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Overwrites UKFCell.motion_model
        """
        return loxodrome.advance(state, t)

    def measurement_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Overwrites UKFCell.measurement_model
        """
        return loxodrome.to_ais(state)

    def error(self, measurement: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Custom error function for measurement and prediction

        The latitudinal and longitudinal positions are expected to be in [-180, +180]. The
        difference d between a pair (a, b) of such values is estimated as:
         - d = a - b if -180 < a - b < +180,
         - d = a - b + 360 if a - b <= -180,
         - d = a - b - 360 if a - b >= 180.

         The course over ground is expected to be in [0, 360]. The difference d between a pair
         (a, b) of such values is estimated as d = ((a - b) mod 360) - 180, where the modulo
         operation does not copy the sign.

        Args:
            measurement: batched measurements
            prediction: batched predictions

        Returns:
            Batches of corrected differences
        """

        def _diff_m180_p180(diff):
            m1 = (diff > -180.)
            m2 = (diff < 180.)
            return diff * (m1 & m2) - (diff + 360.) * ~m1 + (diff - 360.) * ~m2

        def _diff_0_360(diff):
            def _mod(x, n):
                return torch.fmod(torch.fmod(x, n) + n, n)

            return _mod(diff + 180., 360.) - 180.

        dlat = measurement[:, 0] - prediction[:, 0]
        dlon = measurement[:, 1] - prediction[:, 1]
        dsog = measurement[:, 2] - prediction[:, 2]
        dcog = measurement[:, 3] - prediction[:, 3]

        return torch.stack((
            _diff_m180_p180(dlat),
            _diff_m180_p180(dlon),
            dsog,
            _diff_0_360(dcog),
        ), dim=1)


class UKFRNN(KFRNN):
    def __init__(self, *args, **kwargs):
        super(UKFRNN, self).__init__(cell=UKF(*args, **kwargs))

    @property
    def process_noise(self) -> torch.tensor:
        return self.cell.process_noise.data

    @process_noise.setter
    def process_noise(self, data: torch.Tensor) -> None:
        self.cell.process_noise.data = data

    def process_noise_cov(self) -> torch.Tensor:
        return self.cell.process_noise_cov()

    @property
    def measurement_noise(self) -> torch.Tensor:
        return self.cell.measurement_noise.data

    @measurement_noise.setter
    def measurement_noise(self, data: torch.Tensor) -> None:
        self.cell.measurement_noise.data = data

    def measurement_noise_cov(self) -> torch.Tensor:
        return self.cell.measurement_noise_cov()


def init_ukf(*, batch_size: int, debug: bool = True) -> torch.Tensor:
    def _constrain_process_noise(grad):
        """
        Constrain process noise

        Constrains:
         (1)     (0, 0) and (1, 1): common average
         (2)     (2, 2) and (3, 3): common average
         (3) off-diagonal elements: zero

        Index mapping of 4x4 triangular matrix:
        [0 1 2 3 4 5 6 7 8 9] -> [[0 - - -],
                                  [1 2 - -],
                                  [3 4 5 -],
                                  [6 7 8 9]]
        """
        avg_pos = torch.sum(grad[:, (0, 2)], dim=1) / 2.
        avg_vel = torch.sum(grad[:, (5, 9)], dim=1) / 2.
        new_grad = torch.zeros_like(grad)  # (3)
        new_grad[:, 0] = avg_pos  # (1)
        new_grad[:, 2] = avg_pos  # (1)
        new_grad[:, 5] = avg_vel  # (2)
        new_grad[:, 9] = avg_vel  # (2)

        return new_grad

    def _constrain_measurement_noise(grad):
        """
        Constrain measurement noise

        Constrains:
         (1)     (0, 0) and (1, 1): common average
         (2) off-diagonal elements: zero

        Index mapping of 2x2 triangular matrix:
        [0 1 2] -> [[0 -],
                    [1 2]]
        """
        avg = torch.sum(grad[:, (0, 2)], dim=1) / 2.

        new_grad = torch.zeros_like(grad)  # (2)
        new_grad[:, 0] = avg  # (1)
        new_grad[:, 2] = avg  # (1)

        return new_grad

    def _reinit_nans(grad):
        sel = ~torch.isfinite(grad)
        if torch.any(sel):
            new_grad = torch.tensor(grad)
            new_grad[sel] = torch.rand(new_grad[sel].shape)
            print('Warning! Found {torch.sum(sel)} NaN values in gradient')
            return new_grad

    rnn = UKFRNN(batch_size=batch_size, log_cholesky=True)

    rnn.cell.process_noise.register_hook(_reinit_nans)
    rnn.cell.process_noise.register_hook(_constrain_process_noise)

    rnn.cell.measurement_noise.register_hook(_reinit_nans)
    rnn.cell.measurement_noise.register_hook(_constrain_measurement_noise)

    return rnn if debug else torch.jit.script(rnn)
