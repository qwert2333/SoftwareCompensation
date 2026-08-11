# calo/geo.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Geo:
    ecal_cell_size: tuple
    ecal_y_range: tuple
    ecal_xz_window: tuple

    hcal_cell_size: tuple
    hcal_y_range: tuple
    hcal_xz_window: tuple

    inputshape_xz: int

def build_geo(ecal_cell_size_xz: float, ecal_windowsize_xz: float,
              hcal_cell_size_xz: float, hcal_windowsize_xz: float) -> Geo:

    ecal_cell_size = (ecal_cell_size_xz, 7.0, ecal_cell_size_xz)
    ecal_y_range   = (1802.5, 2012.5)
    ecal_xz_window = (ecal_windowsize_xz, ecal_windowsize_xz)

    hcal_cell_size = (hcal_cell_size_xz, 26.2, hcal_cell_size_xz)
    hcal_y_range   = (2082.5, 3342.5)
    hcal_xz_window = (hcal_windowsize_xz, hcal_windowsize_xz)

    inputshape_xz  = int(ecal_windowsize_xz / ecal_cell_size_xz)

    return Geo(
        ecal_cell_size=ecal_cell_size, ecal_y_range=ecal_y_range, ecal_xz_window=ecal_xz_window,
        hcal_cell_size=hcal_cell_size, hcal_y_range=hcal_y_range, hcal_xz_window=hcal_xz_window,
        inputshape_xz=inputshape_xz
    )
