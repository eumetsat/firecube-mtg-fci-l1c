# Copyright 2025-2026 EUMETSAT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sun-Earth distance from an ephemeris."""

from __future__ import annotations

import datetime


def sun_earth_distance_au(time: datetime.datetime) -> float:
    """Return the distance from the Earth's centre to the Sun in AU.

    ``time`` is UTC. Uses astropy's built-in solar ephemeris; no data is
    downloaded.
    """
    # Imported lazily: astropy is only needed when a slot is ingested.
    from astropy.coordinates import get_sun
    from astropy.time import Time
    from astropy.utils import iers

    # Ingest pods may have no network; never fetch Earth-orientation tables.
    with iers.conf.set_temp("auto_download", False):
        return float(get_sun(Time(time, scale="utc")).distance.to_value("AU"))
