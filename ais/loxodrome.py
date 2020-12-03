import math

import torch

from turbokf.turbokf import ekf_cell


class Loxodrome(ekf_cell.EKFCell):
    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, state_size=4, measurement_size=4)

    def _deg2rad(self, deg: torch.Tensor) -> torch.Tensor:
        return deg / 180. * math.pi

    def _kn2deg(self, kn: torch.Tensor) -> torch.Tensor:
        return kn / 60.

    def _kn2rad(self, kn: torch.Tensor) -> torch.Tensor:
        return kn / 60. / 180. * math.pi

    def init_state(self,
                   *,
                   lat_deg: torch.Tensor,
                   lon_deg: torch.Tensor,
                   sog_kn: torch.Tensor,
                   cog_deg: torch.Tensor) -> torch.Tensor:
        """
        Initialize batches of Loxodromes

        A Loxodrome is parametrized as `(lat, lon', vlat, vlon')`, where the first and third
        parameters are the latitudinal position and speed, respectively. The second and fourth
        parameters are the scaled longitudinal position and speed, where the scale factor is the
        cosine of the first parameter (latitudinal position). After scaling, both velocities are
        conserved on Loxodromes through time.

        Args:
            lat_deg: batched latitudes in degree
            lon_deg: batched longitudes in degree
            sog_kn: batched speed over ground in knots
            cog_deg: batched course over ground in degree

        Returns:
            Batches of Loxodromes
        """

        lat_rad = self._deg2rad(lat_deg)
        cog_rad = self._deg2rad(cog_deg)

        vlat_kn = torch.cos(cog_rad) * sog_kn
        scaled_vlon_kn = torch.sin(cog_rad) * sog_kn
        return torch.stack((
            lat_deg,
            (torch.cos(lat_rad) * lon_deg),
            vlat_kn,
            scaled_vlon_kn,
        ), dim=1)

    def motion_model_jacobian(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lat_deg = state[:, 0]
        slon_deg = state[:, 1]
        vlat_kn = state[:, 2]

        vlat_deg = self._kn2deg(vlat_kn)
        lat_rad = self._deg2rad(lat_deg)

        j = torch.zeros(self.batch_size, 4, 4)
        j[:, 0, 2] = 1.
        j[:, 1, 3] = 1.

        j[:, 1, 0] = -slon_deg * vlat_deg / torch.cos(lat_rad) ** 2
        j[:, 1, 1] = -vlat_deg * torch.tan(lat_rad)
        j[:, 1, 2] = -slon_deg * torch.tan(lat_rad)

        return torch.eye(4, 4).unsqueeze(0) + j * t.unsqueeze(1).unsqueeze(2)

    def motion_model(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lat_deg = state[:, 0]
        slon_deg = state[:, 1]
        vlat_kn = state[:, 2]
        svlon_kn = state[:, 3]

        new_lox = torch.empty_like(state)
        new_lox[:, 0] = lat_deg + t * self._kn2deg(vlat_kn)
        new_lox[:, 1] = slon_deg + t * self._kn2deg(svlon_kn) \
                        - t * slon_deg * self._kn2deg(vlat_kn) * torch.tan(self._deg2rad(lat_deg))
        new_lox[:, 2] = vlat_kn
        new_lox[:, 3] = svlon_kn

        return new_lox

    def measurement_model_jacobian(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.eye(4, 4).unsqueeze(0).repeat(self.batch_size, 1, 1)
