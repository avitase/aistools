import math

import torch


def init_from_ais(*,
                  lat_deg: torch.Tensor,
                  lon_deg: torch.Tensor,
                  sog_kn: torch.Tensor,
                  cog_deg: torch.Tensor) -> torch.Tensor:
    """
    Initialize batches of Loxodromes

    A Loxodrome is parametrized as `(lat, lon', vlat, vlon')`, where the first and third parameters
    are the latitudinal position and speed, respectively. The second and fourth parameters
    are the scaled longitudinal position and speed, where the scale factor is the cosine of the
    first parameter (latitudinal position). After scaling, both velocities are approximately
    conserved on Loxodromes through time.

    Args:
        lat: batched latitudes in degree
        lon: batched longitudes in degree
        sog_kn: batched speed over ground in knots
        sog: batched course over ground in degree

    Returns:
        Batches of Loxodromes
    """

    def _deg2rad(deg):
        return deg / 180. * math.pi

    lat_rad = _deg2rad(lat_deg)
    cog_rad = _deg2rad(cog_deg)

    vlat_kn = torch.cos(cog_rad) * sog_kn
    scaled_vlon_kn = torch.sin(cog_rad) * sog_kn
    return torch.stack((lat_deg, torch.cos(lat_rad) * lon_deg, vlat_kn, scaled_vlon_kn), dim=1)


def to_ais(loxodrome: torch.Tensor) -> torch.Tensor:
    """
    Inversion of `init_from_ais`

    Args:
        loxodrome: batched Loxodromes

    Returns:
        Batched tuples of longitude, latitude, speed over ground and course over ground
    """

    def _deg2rad(deg):
        return deg / 180. * math.pi

    def _rad2deg(rad):
        return rad / math.pi * 180.

    lat_deg = loxodrome[:, 0]
    lat_rad = _deg2rad(lat_deg)
    lon_deg = loxodrome[:, 1] / torch.cos(lat_rad)
    vlat_kn = loxodrome[:, 2]
    scaled_vlon = loxodrome[:, 3]

    sog_kn = torch.sqrt(vlat_kn ** 2 + scaled_vlon ** 2)

    cog_rad = torch.atan2(scaled_vlon, vlat_kn)
    cog_deg = _rad2deg(cog_rad)

    return torch.stack((
        lat_deg,
        lon_deg,
        sog_kn,
        cog_deg * (cog_deg > 0.) + (cog_deg + 360.) * (cog_deg < 0.),
    ), dim=1)


def advance(loxodrome: torch.Tensor, *, t: torch.Tensor) -> torch.Tensor:
    """
    Advances batches of Loxodromes in time

    Args:
        loxodrome: batched Loxodromes
        t: batches of time

    Returns:
        Advanced batches of Loxodromes
    """
    lat_deg = loxodrome[:, 0]
    scaled_lon = loxodrome[:, 1]
    vlat_kn = loxodrome[:, 2]
    scaled_vlon = loxodrome[:, 3]

    return torch.stack((
        lat_deg + vlat_kn / 60. * t,
        scaled_lon + scaled_vlon / 60. * t,
        vlat_kn,
        scaled_vlon,
    ), dim=1)
