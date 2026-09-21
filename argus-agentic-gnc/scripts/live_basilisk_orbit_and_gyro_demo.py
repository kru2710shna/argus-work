"""Live 3D Basilisk + Vizard demonstration.

RUN ORDER
---------
1. Open Vizard with the command shown below.
2. Run this file in a second terminal.
3. Vizard shows the satellite moving around Earth live.

OUTPUTS
-------
- Live 3D Vizard simulation
- outputs/live_basilisk_truth_and_gyro.npz
- outputs/live_basilisk_summary.png

THIS DEMO PROVES
----------------
- Basilisk orbit propagation works.
- Basilisk outputs truth position, velocity, attitude, and body rate.
- A synthetic gyro stream can be made from truth + bias + noise.

NOT INCLUDED YET
----------------
- EarthLoc / LightGlue
- FSW-Payload
- LLM agent
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Basilisk.simulation import spacecraft
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
    unitTestSupport,
    vizSupport,
)


def main() -> None:
    # ------------------------------------------------------------
    # 1. Output folder
    # ------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # 2. Basilisk simulation container
    # ------------------------------------------------------------
    sim = SimulationBaseClass.SimBaseClass()

    process = sim.CreateNewProcess("dynamics_process")
    task_name = "dynamics_task"

    # Basilisk updates the spacecraft state every simulated second.
    time_step_ns = macros.sec2nano(1.0)
    process.addTask(sim.CreateNewTask(task_name, time_step_ns))

    # ------------------------------------------------------------
    # 3. Satellite
    # ------------------------------------------------------------
    satellite = spacecraft.Spacecraft()
    satellite.ModelTag = "ArgusCubeSat"
    sim.AddModelToTask(task_name, satellite)

    # ------------------------------------------------------------
    # 4. Earth gravity
    # ------------------------------------------------------------
    gravity_factory = simIncludeGravBody.gravBodyFactory()

    earth = gravity_factory.createEarth()
    earth.isCentralBody = True
    gravity_factory.addBodiesTo(satellite)

    # ------------------------------------------------------------
    # 5. Initial orbit: low Earth orbit
    # ------------------------------------------------------------
    orbit = orbitalMotion.ClassicElements()
    orbit.a = 7_000_000.0              # semi-major axis [m]
    orbit.e = 0.001                    # eccentricity
    orbit.i = 51.6 * macros.D2R        # inclination [rad]
    orbit.Omega = 0.0 * macros.D2R     # RAAN [rad]
    orbit.omega = 0.0 * macros.D2R     # argument of perigee [rad]
    orbit.f = 0.0 * macros.D2R         # true anomaly [rad]

    initial_position_eci_m, initial_velocity_eci_mps = (
        orbitalMotion.elem2rv(earth.mu, orbit)
    )

    satellite.hub.r_CN_NInit = initial_position_eci_m
    satellite.hub.v_CN_NInit = initial_velocity_eci_mps

    # Satellite starts with a small body rotation.
    satellite.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    satellite.hub.omega_BN_BInit = [[0.002], [-0.001], [0.0005]]

    # ------------------------------------------------------------
    # 6. Record Basilisk truth data
    # ------------------------------------------------------------
    simulation_duration_s = 1_200.0  # 20 simulated minutes

    recorder_interval_ns = unitTestSupport.samplingTime(
        macros.sec2nano(simulation_duration_s),
        time_step_ns,
        240,
    )

    truth_recorder = satellite.scStateOutMsg.recorder(recorder_interval_ns)
    sim.AddModelToTask(task_name, truth_recorder)

    # ------------------------------------------------------------
    # 7. Connect Basilisk to Vizard for LIVE 3D visualization
    # ------------------------------------------------------------
    viz = vizSupport.enableUnityVisualization(
        sim,
        task_name,
        satellite,
        liveStream=True,
    )

    viz.settings.mainCameraTarget = "ArgusCubeSat"
    viz.settings.spacecraftCSon = 1
    viz.settings.orbitLinesOn = 1
    viz.settings.trueTrajectoryLinesOn = 1
    viz.settings.showVelocityFrame = 1
    viz.settings.spacecraftSizeMultiplier = 10.0
    viz.settings.ambient = 0.5

    # ------------------------------------------------------------
    # 8. Run
    # ------------------------------------------------------------
    print("Starting live Basilisk simulation.")
    print("Vizard must already be open in DirectComm mode.")

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(simulation_duration_s))
    sim.ExecuteSimulation()

    # ------------------------------------------------------------
    # 9. Read Basilisk truth outputs
    # ------------------------------------------------------------
    time_s = truth_recorder.times() * macros.NANO2SEC
    position_eci_m = truth_recorder.r_BN_N
    velocity_eci_mps = truth_recorder.v_BN_N
    attitude_mrp = truth_recorder.sigma_BN
    body_rate_rad_s = truth_recorder.omega_BN_B

    # ------------------------------------------------------------
    # 10. Create synthetic gyro data
    # Later: replace this with Basilisk sensor output or real IMU.
    # ------------------------------------------------------------
    random = np.random.default_rng(seed=7)

    gyro_bias_rad_s = np.array([2e-5, -1e-5, 3e-5])
    gyro_noise_std_rad_s = 5e-6

    gyro_measurement_rad_s = (
        body_rate_rad_s
        + gyro_bias_rad_s
        + random.normal(
            loc=0.0,
            scale=gyro_noise_std_rad_s,
            size=body_rate_rad_s.shape,
        )
    )

    # Save arrays for the later FSW integration.
    np.savez(
        output_dir / "live_basilisk_truth_and_gyro.npz",
        time_s=time_s,
        position_eci_m=position_eci_m,
        velocity_eci_mps=velocity_eci_mps,
        attitude_mrp=attitude_mrp,
        body_rate_rad_s=body_rate_rad_s,
        gyro_measurement_rad_s=gyro_measurement_rad_s,
        gyro_bias_rad_s=gyro_bias_rad_s,
    )

    # ------------------------------------------------------------
    # 11. Save a small summary plot too
    # ------------------------------------------------------------
    figure = plt.figure(figsize=(12, 5))

    orbit_plot = figure.add_subplot(1, 2, 1, projection="3d")
    orbit_plot.plot(
        position_eci_m[:, 0] / 1e3,
        position_eci_m[:, 1] / 1e3,
        position_eci_m[:, 2] / 1e3,
        label="Basilisk truth orbit",
    )
    orbit_plot.scatter(0, 0, 0, s=80, label="Earth center")
    orbit_plot.set_title("Basilisk truth orbit")
    orbit_plot.set_xlabel("ECI X [km]")
    orbit_plot.set_ylabel("ECI Y [km]")
    orbit_plot.set_zlabel("ECI Z [km]")
    orbit_plot.legend()

    gyro_plot = figure.add_subplot(1, 2, 2)
    for index, axis_name in enumerate(("X", "Y", "Z")):
        gyro_plot.plot(
            time_s,
            body_rate_rad_s[:, index],
            label=f"truth ω{axis_name}",
        )
        gyro_plot.plot(
            time_s,
            gyro_measurement_rad_s[:, index],
            linestyle="--",
            label=f"gyro ω{axis_name}",
        )

    gyro_plot.set_title("Synthetic gyro: truth + bias + noise")
    gyro_plot.set_xlabel("Time [s]")
    gyro_plot.set_ylabel("Angular rate [rad/s]")
    gyro_plot.grid()
    gyro_plot.legend(ncol=2, fontsize=8)

    figure.tight_layout()
    figure.savefig(output_dir / "live_basilisk_summary.png", dpi=180)

    print("PASS: Live Basilisk simulation completed.")
    print(f"Truth position shape: {position_eci_m.shape}")
    print(f"Truth velocity shape: {velocity_eci_mps.shape}")
    print(f"Gyro measurement shape: {gyro_measurement_rad_s.shape}")
    print(f"Data: {output_dir / 'live_basilisk_truth_and_gyro.npz'}")


if __name__ == "__main__":
    main()