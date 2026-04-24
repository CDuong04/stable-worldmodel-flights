import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / 'scripts'
    / 'collect_cartpole_expert_data.py'
)


def load_module():
    module_name = 'collect_cartpole_expert_data'
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so dataclass annotation resolution works.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_state_stays_in_valid_cartpole_ranges():
    module = load_module()
    rng = np.random.default_rng(0)

    for profile in module.PROFILES:
        for _ in range(25):
            state = module.sample_state(profile, rng)
            if state is None:
                continue

            x, theta, x_dot, theta_dot = state
            assert -1.7 <= x <= 1.7
            assert -np.pi <= theta <= np.pi
            assert -9.0 <= x_dot <= 9.0
            assert -12.0 <= theta_dot <= 12.0


def test_sample_variation_values_stay_within_declared_bounds():
    module = load_module()
    rng = np.random.default_rng(1)

    for profile in module.PROFILES:
        values, watch_keys = module.sample_variation_values(
            rng,
            vary_visuals=True,
            vary_dynamics=True,
            profile=profile,
        )
        assert set(watch_keys) == {
            'agent.color',
            'floor.color',
            'light.intensity',
            'agent.cart_shape',
            'agent.cart_mass',
            'agent.pole_density',
            'gravity.x',
            'gravity.y',
            'gravity.z',
        }
        assert all(0.0 <= c <= 1.0 for c in values['agent.color'])
        assert values['agent.cart_shape'] in (0, 1)
        assert 0.5 <= values['agent.cart_mass'][0] <= 1.5
        assert 500.0 <= values['agent.pole_density'][0] <= 1500.0
        assert -5.0 <= values['gravity.x'][0] <= 5.0
        assert -5.0 <= values['gravity.y'][0] <= 5.0
        assert -20.0 <= values['gravity.z'][0] <= 0.0


def test_sample_episode_spec_returns_valid_profile_and_options():
    module = load_module()
    rng = np.random.default_rng(2)
    args = type(
        'Args',
        (),
        {'vary_visuals': True, 'vary_dynamics': True},
    )()

    seen = set()
    for _ in range(200):
        profile, options = module.sample_episode_spec(args, rng)
        seen.add(profile.name)
        assert profile in module.PROFILES
        assert options is None or isinstance(options, dict)
        if options is not None and 'state' in options:
            assert len(options['state']) == 4

    assert len(seen) >= 4
