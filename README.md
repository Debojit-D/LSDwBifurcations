# Learning Dynamical Systems with Bifurcations

Course project based on:

- Farshad Khadivar, Ilaria Lauzana, and Aude Billard
- *Learning Dynamical Systems with Bifurcations*

This project studies how a single parameterized dynamical system can smoothly transition between:

- a **stable point attractor** for reaching / pick motion
- a **stable limit cycle** for periodic motion such as polishing

A MuJoCo simulation of the Franka robot is used to demonstrate the behavior in a robotics setting.

## Demo

Final video: https://youtu.be/7tVdRKwNmac

## Key idea

The main idea is to avoid hard switching between separate controllers. Instead, one interpretable parameter changes the system behavior continuously.

The notebook [main.ipynb](main.ipynb) contains:

- stability and Lyapunov-style verification
- phase portrait analysis
- limit-cycle and bifurcation comparisons
- numerical checks for convergence and radius error

## Robot simulation

The MuJoCo robot tasks use controller utilities from:

- [`utils.mj_velocity_control.JointVelocityController`](utils/mj_velocity_control/mj_velocity_ctrl.py)
- [`utils.dls_velocity_control.DLSVelocityPlanner`](utils/dls_velocity_control/dls_velocity_ctrl.py)
- [`utils.bifurcation.MP4Recorder`](utils/bifurcation/mujoco_recording.py)

Task scripts include:

- [scripts/test_bifurcationV0.py](scripts/test_bifurcationV0.py)
- [scripts/test_bifurcationV1.py](scripts/test_bifurcationV1.py)
- [scripts/test_without_bifurcation.py](scripts/test_without_bifurcation.py)
- [scripts/test_robot_trace_circle.py](scripts/test_robot_trace_circle.py)

## Repository structure

- [main.ipynb](main.ipynb) — theory, plots, and numerical verification
- [scripts/](scripts/) — MuJoCo task demos and tests
- [utils/](utils/) — controller and bifurcation helpers
- [robot_descriptions/](robot_descriptions/) — Franka XML models and assets

## Dependencies

Typical dependencies used in this workspace:

- `numpy`
- `scipy`
- `mujoco`
- `matplotlib`

Optional recording backends:

- `imageio[ffmpeg]`
- `opencv-python`
- system `ffmpeg`

## Usage

Run the notebook for the analytical results:

```bash
jupyter notebook main.ipynb
```

Run a task script, for example:

```bash
python scripts/test_bifurcationV1.py
```

## Notes

The Franka robot assets are distributed under the license included in:

- [robot_descriptions/franka/LICENSE](robot_descriptions/franka/LICENSE)
- [robot_descriptions/franka_emika_panda/LICENSE](robot_descriptions/franka_emika_panda/LICENSE)

## Results

The project shows that:

- the same dynamical system can represent both point attraction and periodic motion
- smooth parameter variation avoids abrupt controller switching
- the limit-cycle radius converges with small numerical error in simulation