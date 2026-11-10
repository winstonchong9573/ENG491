
"""
VLEO NRLMSISE-00 density and Thermal Desktop FMH generator.

Install:
    pip install numpy matplotlib astropy nrlmsise00

Run:
    python vleo_orbit_to_td.py

Outputs:
    vleo_density_300km.csv
    thermal_desktop_fmh.csv
    thermal_desktop_fmh_paste.txt
    vleo_density_300km.png

Orbit inputs are intended to match a Thermal Desktop Keplerian orbit.
The model uses two-body Keplerian propagation, with no J2, drag,
or other perturbations.

GCRS - Geocentric Celestial Reference System
ITRS - International Terrestrial Reference System

Velocity and Sun/Planet vectors are exported in GCRS axes.
Verify that these match Thermal Desktop's expected coordinate frame.

F10.7 - Solar radio flux at 10.7 cm wavelength
Ap - Daily geomagnetic activity index
These two parameters need to be looked up from space weather data sources for accurate simulation results. Dependent on the day of the simulation
Look up here: https://celestrak.org/SpaceData/

ALPHA - Effective energy accommodation / heating coefficient

"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation,
    EarthLocation,
    GCRS,
    ITRS,
    get_sun,
)
from astropy.time import Time

from nrlmsise00 import msise_model


# ============================================================
# 1. ORBIT DEFINITION — COPY FROM THERMAL DESKTOP
# ============================================================

START_UTC = datetime(
    2026, 10, 20, 12, 0, 0,
    tzinfo=timezone.utc
)

EARTH_RADIUS_KM = 6378.14
EARTH_MU_KM3_S2 = 398601.0
SIDEREAL_PERIOD_S = 86164.1

# If SEMI_MAJOR_AXIS_KM is None, use:
# a = EARTH_RADIUS_KM + ALTITUDE_KM

ALTITUDE_KM = 200.0
SEMI_MAJOR_AXIS_KM = None

ECCENTRICITY = 0.0
INCLINATION_DEG = 0.0
RAAN_DEG = 0.0
ARGUMENT_OF_PERIGEE_DEG = 0.0
TRUE_ANOMALY_AT_EPOCH_DEG = 0.0

# None = one calculated orbital period
DURATION_S = None

STEP_S = 60.0


# ============================================================
# 2. ATMOSPHERIC MODEL
# ============================================================

# Demonstration values, not observed space weather.
# For historical runs, replace with actual F10.7 and Ap data.

F107_PREVIOUS_DAY = 120.0
F107_81_DAY_AVERAGE = 120.0
AP_DAILY = 4.0

# Effective energy accommodation / heating coefficient
ALPHA = 1.0

# "velocity" = spacecraft velocity relative to atmosphere
# "incoming" = opposite of that direction
VELOCITY_VECTOR_CONVENTION = "velocity"


# ============================================================
# 3. OUTPUT SETTINGS
# ============================================================

OUTPUT_DENSITY = Path(f"vleo_density_{ALTITUDE_KM:g}km.csv")
OUTPUT_FMH = Path("thermal_desktop_fmh.csv")
OUTPUT_PASTE = Path("thermal_desktop_fmh_paste.txt")
OUTPUT_PLOT = Path(f"vleo_density_{ALTITUDE_KM:g}km.png")


# ============================================================
# 4. ORBITAL PARAMETERS
# ============================================================

def orbital_parameters():

    a = (
        SEMI_MAJOR_AXIS_KM
        if SEMI_MAJOR_AXIS_KM is not None
        else EARTH_RADIUS_KM + ALTITUDE_KM
    )

    e = ECCENTRICITY

    if a <= 0:
        raise ValueError("Semi-major axis must be positive.")

    if not (0 <= e < 1):
        raise ValueError("Requires an elliptical orbit: 0 <= e < 1.")

    period = 2 * np.pi * np.sqrt(
        a**3 / EARTH_MU_KM3_S2
    )

    return a, e, period


# ============================================================
# 5. SOLVE KEPLER'S EQUATION
# ============================================================

def solve_kepler(mean_anomaly, eccentricity):

    E = mean_anomaly

    for _ in range(50):

        f = E - eccentricity * np.sin(E) - mean_anomaly
        fp = 1 - eccentricity * np.cos(E)

        correction = f / fp
        E -= correction

        if abs(correction) < 1e-12:
            break

    return E


# ============================================================
# 6. ORBIT POSITION AND VELOCITY
# ============================================================

def orbit_state(t_s, a, e):

    inc = np.deg2rad(INCLINATION_DEG)
    raan = np.deg2rad(RAAN_DEG)
    argp = np.deg2rad(ARGUMENT_OF_PERIGEE_DEG)
    nu0 = np.deg2rad(TRUE_ANOMALY_AT_EPOCH_DEG)

    mu = EARTH_MU_KM3_S2

    # Initial eccentric anomaly
    E0 = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(nu0 / 2),
        np.sqrt(1 + e) * np.cos(nu0 / 2)
    )

    M0 = E0 - e * np.sin(E0)

    mean_motion = np.sqrt(mu / a**3)

    M = M0 + mean_motion * t_s
    M = (M + np.pi) % (2 * np.pi) - np.pi

    E = solve_kepler(M, e)

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    p = a * (1 - e**2)

    r = p / (1 + e * np.cos(nu))

    # Position in perifocal frame (km)
    r_pf = np.array([
        r * np.cos(nu),
        r * np.sin(nu),
        0.0
    ])

    # Velocity in perifocal frame (km/s)
    v_pf = np.sqrt(mu / p) * np.array([
        -np.sin(nu),
        e + np.cos(nu),
        0.0
    ])

    # Rotation matrices
    def Rz(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    def Rx(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ])

    rotation = (
        Rz(raan)
        @ Rx(inc)
        @ Rz(argp)
    )

    position_km = rotation @ r_pf
    velocity_km_s = rotation @ v_pf

    return position_km, velocity_km_s


# ============================================================
# 7. CONVERT POSITION TO GEODETIC COORDINATES
# ============================================================

def geodetic_position(position_km, date_utc):

    obstime = Time(date_utc)

    # Interpret propagated inertial coordinates as GCRS.
    position_gcrs = GCRS(
        CartesianRepresentation(
            position_km * u.km
        ),
        obstime=obstime
    )

    # Earth-fixed position
    position_itrs = position_gcrs.transform_to(
        ITRS(obstime=obstime)
    )

    location = EarthLocation.from_geocentric(
        position_itrs.cartesian.x,
        position_itrs.cartesian.y,
        position_itrs.cartesian.z
    )

    lon, lat, height = location.to_geodetic()

    return (
        lat.to_value(u.deg),
        lon.to_value(u.deg),
        height.to_value(u.km)
    )


# ============================================================
# 8. ATMOSPHERE-RELATIVE VELOCITY
# ============================================================

def relative_velocity(position_km, velocity_km_s):

    # Convert to SI units
    r = position_km * 1000.0
    v = velocity_km_s * 1000.0

    omega = np.array([
        0.0,
        0.0,
        2 * np.pi / SIDEREAL_PERIOD_S
    ])

    # Simplified rigidly co-rotating atmosphere
    v_atmosphere = np.cross(omega, r)

    v_relative = v - v_atmosphere

    speed = np.linalg.norm(v_relative)

    direction = v_relative / speed

    if VELOCITY_VECTOR_CONVENTION == "incoming":
        direction = -direction
    elif VELOCITY_VECTOR_CONVENTION != "velocity":
        raise ValueError("Unknown velocity vector convention.")

    return direction, speed


# ============================================================
# 9. SUN AND PLANET DIRECTIONS
# ============================================================

def sun_planet_vectors(position_km, date_utc):

    obstime = Time(date_utc)

    # Sun geocentric position in GCRS
    sun = get_sun(obstime).transform_to(
        GCRS(obstime=obstime)
    )

    sun_position_km = sun.cartesian.xyz.to_value(u.km)

    # Direction from spacecraft to Sun
    sun_vector = sun_position_km - position_km
    sun_vector /= np.linalg.norm(sun_vector)

    # Direction from spacecraft toward Earth center
    planet_vector = -position_km
    planet_vector /= np.linalg.norm(planet_vector)

    return sun_vector, planet_vector


# ============================================================
# 10. NRLMSISE-00 DENSITY
# ============================================================

def atmospheric_density(date_utc, altitude_km, lat, lon):

    densities, temperatures = msise_model(
        date_utc.replace(tzinfo=None),
        float(altitude_km),
        float(lat),
        float(lon),
        F107_81_DAY_AVERAGE,
        F107_PREVIOUS_DAY,
        AP_DAILY,
        method="gtd7d"
    )

    # g/cm^3 -> kg/m^3
    rho = densities[5] * 1000.0

    exospheric_temp = temperatures[0]
    local_temp = temperatures[1]

    return rho, exospheric_temp, local_temp


# ============================================================
# 11. MAIN SIMULATION
# ============================================================

def main():

    a, e, period = orbital_parameters()

    duration = period if DURATION_S is None else DURATION_S

    print(f"Semi-major axis: {a:.3f} km")
    print(f"Eccentricity: {e:.6f}")
    print(f"Inclination: {INCLINATION_DEG:.3f} deg")
    print(f"RAAN: {RAAN_DEG:.3f} deg")
    print(f"Orbital period: {period:.2f} s")
    print(f"Duration: {duration:.2f} s")

    if STEP_S <= 0 or duration <= 0:
        raise ValueError("STEP_S and duration must be positive.")

    times = np.arange(0, duration, STEP_S)

    if len(times) == 0 or not np.isclose(times[-1], duration):
        times = np.append(times, duration)

    density_rows = []
    fmh_rows = []

    for t in times:

        date_utc = START_UTC + timedelta(seconds=float(t))

        # Propagate orbit
        position_km, velocity_km_s = orbit_state(t, a, e)

        # Ground track and geodetic altitude
        lat, lon, altitude = geodetic_position(
            position_km,
            date_utc
        )

        # Atmospheric density
        rho, texo, tlocal = atmospheric_density(
            date_utc,
            altitude,
            lat,
            lon
        )

        # Relative velocity
        direction, speed = relative_velocity(
            position_km,
            velocity_km_s
        )

        vx, vy, vz = direction

        # Free molecular heating flux
        flux = 0.5 * ALPHA * rho * speed**3

        # Sun / Earth direction vectors
        sun, planet = sun_planet_vectors(
            position_km,
            date_utc
        )

        # Diagnostic output
        density_rows.append([
            t,
            date_utc.isoformat(),
            altitude,
            lat,
            lon,
            rho,
            position_km[0],
            position_km[1],
            position_km[2],
            velocity_km_s[0],
            velocity_km_s[1],
            velocity_km_s[2],
            vx,
            vy,
            vz,
            speed,
            flux,
            texo,
            tlocal
        ])

        # Thermal Desktop FMH table
        fmh_rows.append([
            f"{t:.2f}",
            f"{vx:.8f}",
            f"{vy:.8f}",
            f"{vz:.8f}",
            f"{flux:.6f}",
            f"{sun[0]:.8f}",
            f"{sun[1]:.8f}",
            f"{sun[2]:.8f}",
            f"{planet[0]:.8f}",
            f"{planet[1]:.8f}",
            f"{planet[2]:.8f}"
        ])

    # ========================================================
    # 12. SAVE DENSITY CSV
    # ========================================================

    density_header = [
        "seconds",
        "utc",
        "altitude_km",
        "latitude_deg",
        "longitude_deg",
        "density_kg_m3",
        "position_x_km",
        "position_y_km",
        "position_z_km",
        "orbital_vx_km_s",
        "orbital_vy_km_s",
        "orbital_vz_km_s",
        "fmh_direction_x",
        "fmh_direction_y",
        "fmh_direction_z",
        "relative_speed_m_s",
        "fmh_flux_W_m2",
        "exospheric_temp_K",
        "local_temp_K"
    ]

    with OUTPUT_DENSITY.open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow(density_header)
        writer.writerows(density_rows)

    # ========================================================
    # 13. SAVE THERMAL DESKTOP CSV
    # ========================================================

    fmh_header = [
        "Time [s]",
        "Vx", "Vy", "Vz",
        "Flux [W/m^2]",
        "Sun X", "Sun Y", "Sun Z",
        "Planet X", "Planet Y", "Planet Z"
    ]

    with OUTPUT_FMH.open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow(fmh_header)
        writer.writerows(fmh_rows)

    # Paste-ready rows without header
    with OUTPUT_PASTE.open(
        "w", encoding="utf-8"
    ) as f:

        for row in fmh_rows:
            f.write("\t".join(row) + "\n")

    # ========================================================
    # 14. PLOT RESULTS
    # ========================================================

    seconds = np.array([
        row[0] for row in density_rows
    ])

    densities = np.array([
        row[5] for row in density_rows
    ])

    fluxes = np.array([
        row[16] for row in density_rows
    ])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    axes[0].plot(seconds / 60, densities)
    axes[0].set_ylabel("Density (kg/m^3)")
    axes[0].set_title("NRLMSISE-00 Atmospheric Density")
    axes[0].grid(True)
    axes[0].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0)
    )

    axes[1].plot(seconds / 60, fluxes)
    axes[1].set_xlabel("Time since orbit start (min)")
    axes[1].set_ylabel("FMH Flux (W/m^2)")
    axes[1].set_title("Free Molecular Heating Flux")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=170)
    plt.show()

    # ========================================================
    # 15. SUMMARY
    # ========================================================

    print()
    print(f"Density range: "
          f"{densities.min():.3e} to "
          f"{densities.max():.3e} kg/m^3")

    print(f"FMH flux range: "
          f"{fluxes.min():.3f} to "
          f"{fluxes.max():.3f} W/m^2")

    print(f"Number of rows: {len(fmh_rows)}")

    print()
    print("Created files:")
    print(OUTPUT_DENSITY.resolve())
    print(OUTPUT_FMH.resolve())
    print(OUTPUT_PASTE.resolve())
    print(OUTPUT_PLOT.resolve())


if __name__ == "__main__":
    main()
