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

    lat = _deg2rad(lat_deg)
    lon = _deg2rad(lon_deg)
    sog = _deg2rad(sog_kn / 60.)
    cog = _deg2rad(cog_deg)

    vlat = torch.cos(cog) * sog
    vlon = torch.sin(cog) / torch.cos(lat) * sog
    lon_scale = torch.cos(lat)
    return torch.stack((lat, lon_scale * lon, vlat, lon_scale * vlon), dim=1)


def to_ais(loxodrome: torch.Tensor) -> torch.Tensor:
    """
    Inversion of `init_from_ais`

    Args:
        loxodrome: batched Loxodromes

    Returns:
        Batched tuples of longitude, latitude, speed over ground and course over ground
    """

    def _rad2deg(rad):
        return rad / math.pi * 180.

    lat = loxodrome[:, 0]
    lon = loxodrome[:, 1] / torch.cos(lat)
    vlat = loxodrome[:, 2]
    scaled_vlon = loxodrome[:, 3]

    sog = torch.sqrt(vlat ** 2 + scaled_vlon ** 2)
    cog = torch.atan2(scaled_vlon, vlat)

    lat_deg = _rad2deg(lat)
    lon_deg = _rad2deg(lon)
    sog_kn = _rad2deg(sog) * 60.
    cog_deg = _rad2deg(cog)

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
    lat = loxodrome[:, 0]
    scaled_lon = loxodrome[:, 1]
    vlat = loxodrome[:, 2]
    scaled_vlon = loxodrome[:, 3]

    return torch.stack((
        lat + vlat * t,
        scaled_lon + scaled_vlon * t,
        vlat,
        scaled_vlon,
    ), dim=1)
