import math

import torch

from turbokf.turbokf import ekf_cell


class Loxodrome(ekf_cell.EKFCell):
    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, state_size=4, measurement_size=4)

    def _deg2rad(self, deg: torch.Tensor) -> torch.Tensor:
        return deg / 180. * math.pi

    def _rad2deg(self, rad: torch.Tensor) -> torch.Tensor:
        return rad / math.pi * 180.

    def _kn2rad(self, kn: torch.Tensor) -> torch.Tensor:
        return self._deg2rad(kn) / 60.

    def _rad2kn(self, rad: torch.Tensor) -> torch.Tensor:
        return self._rad2deg(rad) * 60.

    def _wrap_mpi_ppi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wraps x onto the interval [-pi, pi] by adding / subtracting 2pi

        Args:
            x: angle in radians

        Returns:
            x       if x in [-pi, +pi]
            x + 2pi if x < -pi
            x - 2pi if x > +pi
        """
        m1 = (x > -math.pi)
        m2 = (x < math.pi)
        return x * (m1 & m2) + (x + 2. * math.pi) * ~m1 + (x - 2 * math.pi) * ~m2

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
        lon_rad = self._deg2rad(lon_deg)
        cog_rad = self._deg2rad(cog_deg)
        sog_rad = self._kn2rad(sog_kn)

        vlat_rad = torch.cos(cog_rad) * sog_rad
        vlon_rad = torch.sin(cog_rad) * sog_rad / torch.cos(lat_rad)
        return torch.stack((
            lat_rad,
            lon_rad,
            vlat_rad,
            vlon_rad,
        ), dim=1)

    def motion_model_jacobian(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lat = state[:, 0]
        vlat = state[:, 2]
        vlon = state[:, 3]

        sec_lat = 1. / torch.cos(lat)
        tan_lat = torch.tan(lat)

        j = torch.zeros(self.batch_size, 4, 4)
        j[:, 0, 2] = 1.
        j[:, 1, 3] = 1.

        j[:, 3, 0] = vlat * vlon * sec_lat ** 2
        j[:, 3, 2] = vlon * tan_lat
        j[:, 3, 3] = vlat * tan_lat

        return torch.eye(4, 4).unsqueeze(0) + j * t.unsqueeze(1).unsqueeze(2)

    def motion_model(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lat = state[:, 0]
        lon = state[:, 1]
        vlat = state[:, 2]
        vlon = state[:, 3]

        new_lox = torch.empty_like(state)
        new_lox[:, 0] = lat + t * vlat
        new_lox[:, 1] = self._wrap_mpi_ppi(lon + t * vlon)
        new_lox[:, 2] = vlat
        new_lox[:, 3] = vlon * (1 + t * vlat * torch.tan(lat))

        return new_lox

    def measurement_model_jacobian(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.eye(4, 4).unsqueeze(0).repeat(self.batch_size, 1, 1)

    def innovation(self, measurement: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        dlat = measurement[:, 0] - prediction[:, 0]
        dlon = measurement[:, 1] - prediction[:, 1]
        dvlat = measurement[:, 2] - prediction[:, 2]
        dvlon = measurement[:, 3] - prediction[:, 3]

        return torch.stack((
            dlat,
            self._wrap_mpi_ppi(dlon),
            dvlat,
            dvlon,
        ), dim=1)
