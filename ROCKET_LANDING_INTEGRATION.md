# Rocket Landing Environment Integration

This document describes the integration of PyFlyt's Rocket Landing environment into the stable-worldmodel framework.

## Overview

The RocketLanding environment simulates a rocket landing task where the agent must control a rocket falling at terminal velocity to land safely on a landing pad with limited fuel. This integration exposes controllable variations such as initial position, velocity, orientation, and fuel levels, making it ideal for world model research.

## Installation

### 1. Install stable-worldmodel

```bash
pip install uv
uv pip install -e .
```

### 2. Install PyFlyt

```bash
pip install PyFlyt
```

For GPU-accelerated rendering (optional but recommended):
```bash
pip install PyFlyt[viz]
```

### 3. Verify Installation

Run the quick test to verify everything is working:

```bash
python examples/quick_test_rocket.py
```

If all tests pass, you're ready to go!

## Quick Start

### Basic Usage

```python
import stable_worldmodel as swm

# Create the world with rocket landing environment
world = swm.World(
    "swm/RocketLanding-v0",
    num_envs=4,
    image_shape=(224, 224),
    render_mode="rgb_array",
)

# Collect data with random policy
world.set_policy(swm.policy.RandomPolicy())
world.record_dataset(
    "rocket-landing-data",
    episodes=100,
    seed=2347,
    options={
        "variation": (
            "rocket.start_pos",
            "rocket.start_vel",
            "rocket.start_orn",
        )
    },
)
```

### Running the Full Example

The complete example demonstrates data collection, video recording, and evaluation:

```bash
python examples/example_rocket_landing.py
```

This script will:
1. Create the RocketLanding environment
2. Display available variations
3. Collect training data with a random policy
4. Record sample videos
5. Evaluate the random policy

## Variation Space

The RocketLanding environment exposes the following controllable variations:

### Rocket Variations

| Variation | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `rocket.start_pos` | Box(3,) | x: [-50, 50] m<br>y: [-50, 50] m<br>z: [100, 400] m | [0, 0, 300] | Initial position in 3D space |
| `rocket.start_vel` | Box(3,) | vx: [-20, 20] m/s<br>vy: [-20, 20] m/s<br>vz: [-100, 0] m/s | [0, 0, -50] | Initial velocity (falling) |
| `rocket.start_orn` | Box(3,) | roll: [-0.5, 0.5] rad<br>pitch: [-0.5, 0.5] rad<br>yaw: [-π, π] rad | [0, 0, 0] | Initial orientation (Euler angles) |
| `rocket.starting_fuel_ratio` | Box() | [0.01, 0.2] | 0.05 | Fuel level (1% to 20%) |
| `rocket.start_ang_vel` | Box(3,) | [-1, 1] rad/s | [0, 0, 0] | Initial angular velocity |

### Landing Pad Variations

| Variation | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `landing_pad.position` | Box(2,) | x: [-20, 20] m<br>y: [-20, 20] m | [0, 0] | Landing pad position |

## Action Space

The action space is a 7-dimensional continuous space controlling:

1. **Finlet X** (index 0): Fin deflection in X axis
2. **Finlet Y** (index 1): Fin deflection in Y axis
3. **Finlet Roll** (index 2): Fin roll control
4. **Booster Ignition** (index 3): Engine on/off
5. **Throttle** (index 4): Engine power level
6. **Booster Gimbal X** (index 5): Engine gimbal X axis
7. **Booster Gimbal Y** (index 6): Engine gimbal Y axis

## Observation Space

The observation space depends on the angle representation:

- **Euler mode** (default): 20-dimensional state vector
- **Quaternion mode**: 21-dimensional state vector

The observation includes:
- Position (x, y, z)
- Velocity (vx, vy, vz)
- Orientation (roll, pitch, yaw or quaternion)
- Angular velocity
- Fuel level
- Distance to landing pad
- Other relevant state information

## Usage Examples

### Example 1: Collect Data with Varying Initial Conditions

```python
import stable_worldmodel as swm

world = swm.World(
    "swm/RocketLanding-v0",
    num_envs=8,
    image_shape=(224, 224),
    render_mode="rgb_array",
)

world.set_policy(swm.policy.RandomPolicy(seed=42))

# Vary initial position, velocity, and orientation
world.record_dataset(
    "rocket-varied-conditions",
    episodes=500,
    seed=2347,
    options={
        "variation": (
            "rocket.start_pos",
            "rocket.start_vel",
            "rocket.start_orn",
            "rocket.starting_fuel_ratio",
        )
    },
)
```

### Example 2: Evaluate a Policy

```python
import stable_worldmodel as swm

world = swm.World(
    "swm/RocketLanding-v0",
    num_envs=4,
    image_shape=(224, 224),
    render_mode="rgb_array",
)

world.set_policy(swm.policy.RandomPolicy())

results = world.evaluate(
    episodes=100,
    seed=5555,
    options={
        "variation": ("rocket.start_pos", "rocket.start_vel")
    },
)

print(f"Success Rate: {results['success_rate']:.1f}%")
```

### Example 3: Record Videos

```python
import stable_worldmodel as swm

world = swm.World(
    "swm/RocketLanding-v0",
    num_envs=4,
    image_shape=(224, 224),
    render_mode="rgb_array",
)

world.set_policy(swm.policy.RandomPolicy())

world.record_video(
    "./rocket_videos",
    max_steps=500,
    fps=30,
    seed=9999,
    options={
        "variation": (
            "rocket.start_pos",
            "rocket.start_vel",
            "rocket.start_orn",
        )
    },
)
```

## Next Steps

This is a proof-of-concept integration. Future enhancements include:

### 1. Train a World Model

Once you've collected sufficient data, train a world model:

```python
import stable_worldmodel as swm

# Train a world model (e.g., DINOWM)
swm.pretraining(
    "scripts/train/dinowm.py",
    dataset_name="rocket-landing-data",
    output_model_name="rocket_dinowm",
    dump_object=True,
)
```

### 2. Implement Expert Policy

For better data collection, implement an expert policy that can successfully land the rocket. This could be:
- A PID controller
- A model-based controller
- An RL policy trained on the task
- A scripted policy that uses domain knowledge

### 3. Model-Based Planning

Use the trained world model with a planner:

```python
import stable_worldmodel as swm

# Load trained world model
model = swm.policy.AutoCostModel("rocket_dinowm")

# Configure planning
config = swm.PlanConfig(
    horizon=20,
    receding_horizon=10,
    action_block=2,
)

# Create solver
solver = swm.solver.GDSolver(model, n_steps=20)

# Create policy
policy = swm.policy.WorldModelPolicy(solver=solver, config=config)

# Evaluate
world.set_policy(policy)
results = world.evaluate(episodes=50, seed=2347)
```

### 4. Advanced Integration

For more control over the PyFlyt environment:
- Expose additional PyFlyt parameters (control frequency, physics Hz)
- Add custom goal generation (e.g., different landing scenarios)
- Implement wind/weather variations
- Add obstacle avoidance scenarios

### 5. FlightGear Integration (Future)

Once the rocket landing integration is mature, similar patterns can be used to integrate more complex simulators like FlightGear.

## Troubleshooting

### PyFlyt Import Error

```
ImportError: PyFlyt is required for the RocketLanding environment
```

**Solution**: Install PyFlyt with `pip install PyFlyt`

### Rendering Issues

If rendering fails or produces black frames:

1. Ensure you have a display or are using a virtual display (e.g., Xvfb)
2. Try installing the visualization dependencies: `pip install PyFlyt[viz]`
3. Check your OpenGL drivers are up to date

### Slow Performance

If the environment is running slowly:

1. Reduce `num_envs` (fewer parallel environments)
2. Reduce `image_shape` (smaller rendered images)
3. Use fewer `episodes` during testing
4. Ensure GPU acceleration is available for PyFlyt

### Environment Not Registered

```
gymnasium.error.NameNotFound: Environment swm/RocketLanding-v0 doesn't exist
```

**Solution**: Make sure you've installed the package with `pip install -e .` from the repo root

## File Structure

```
stable_worldmodel/
├── envs/
│   └── rocket_landing.py          # RocketLanding environment wrapper
├── __init__.py                     # Registration of RocketLanding-v0
examples/
├── quick_test_rocket.py            # Quick verification test
└── example_rocket_landing.py       # Full demonstration example
```

## References

- [PyFlyt Documentation](https://taijunjet.com/PyFlyt/)
- [PyFlyt Rocket Landing Environment](https://taijunjet.com/PyFlyt/documentation/gym_envs/rocket_landing_env.html)
- [PyFlyt Rocket Class](https://taijunjet.com/PyFlyt/documentation/core/drones/rocket.html)
- [stable-worldmodel Repository](https://github.com/rbalestr-lab/stable-worldmodel)

## Citation

If you use this integration in your research, please cite both stable-worldmodel and PyFlyt.

## License

This integration follows the MIT License of the stable-worldmodel project.
