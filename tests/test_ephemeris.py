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

"""Numerics tests for the Sun-Earth distance."""

from __future__ import annotations

import datetime

import pytest

from firecube_mtg_fci_l1c._ephemeris import sun_earth_distance_au


# Earth-centre to Sun distance derived from the state tables of three MTI1
# FDHSI products (Sun-satellite distance, satellite position and subsolar
# point), i.e. independent of astropy.
@pytest.mark.unit
@pytest.mark.parametrize(
    ("time", "product_derived_au"),
    [
        (datetime.datetime(2025, 6, 15, 12, 14, 40), 1.0157259),
        (datetime.datetime(2026, 9, 28, 12, 24, 30), 1.0020210),
        (datetime.datetime(2026, 9, 29, 0, 14, 30), 1.0018826),
    ],
)
def test_sun_earth_distance_matches_fci_product_geometry(
    time: datetime.datetime, product_derived_au: float
) -> None:
    # 2e-7 AU is 30 km; the product values are float32 with 16 km resolution.
    assert sun_earth_distance_au(time) == pytest.approx(product_derived_au, abs=2e-7)


@pytest.mark.unit
def test_sun_earth_distance_spans_perihelion_to_aphelion() -> None:
    # Published values for 2026: perihelion 3 January, aphelion 6 July.
    perihelion = sun_earth_distance_au(datetime.datetime(2026, 1, 3, 17, 16))
    aphelion = sun_earth_distance_au(datetime.datetime(2026, 7, 6, 17, 31))

    assert perihelion == pytest.approx(0.98330, abs=1e-5)
    assert aphelion == pytest.approx(1.01664, abs=1e-5)
