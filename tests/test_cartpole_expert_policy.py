import numpy as np

from stable_worldmodel.envs.dmcontrol import CartpoleExpertPolicy


def test_cartpole_expert_policy_returns_vectorized_actions():
    policy = CartpoleExpertPolicy(seed=0)

    info = {
        'qpos': np.array(
            [
                [0.0, np.pi],
                [0.05, 0.08],
            ],
            dtype=np.float32,
        ),
        'qvel': np.array(
            [
                [0.0, 1.2],
                [0.1, -0.2],
            ],
            dtype=np.float32,
        ),
    }

    action = policy.get_action(info)

    assert action.shape == (2, 1)
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)


def test_cartpole_expert_policy_uses_balance_controller_near_upright():
    policy = CartpoleExpertPolicy(
        balance_k_theta=10.0,
        balance_k_theta_dot=0.0,
        balance_k_x=0.0,
        balance_k_x_dot=0.0,
        noise_std=0.0,
        force_scale=1.0,
        seed=0,
    )

    info = {
        'qpos': np.array([[0.0, 0.1]], dtype=np.float32),
        'qvel': np.array([[0.0, 0.0]], dtype=np.float32),
    }

    action = policy.get_action(info)

    assert action[0, 0] > 0.0


def test_cartpole_expert_policy_supports_single_state_inputs():
    policy = CartpoleExpertPolicy(seed=123)

    info = {
        'qpos': np.array([0.0, np.pi], dtype=np.float32),
        'qvel': np.array([0.0, 0.5], dtype=np.float32),
    }

    action = policy.get_action(info)

    assert action.shape == (1, 1)


def test_cartpole_expert_policy_kicks_from_hanging_rest():
    policy = CartpoleExpertPolicy(
        noise_std=0.0,
        burst_prob=0.0,
        seed=0,
    )

    info = {
        'qpos': np.array([[0.0, np.pi]], dtype=np.float32),
        'qvel': np.array([[0.0, 0.0]], dtype=np.float32),
    }

    action = policy.get_action(info)

    assert abs(action[0, 0]) > 0.5


def test_cartpole_expert_policy_pushes_opposite_the_downward_lean():
    policy = CartpoleExpertPolicy(
        noise_std=0.0,
        burst_prob=0.0,
        seed=0,
    )

    # Slight right lean while hanging down should push cart left.
    info = {
        'qpos': np.array([[0.0, np.pi - 0.1]], dtype=np.float32),
        'qvel': np.array([[0.0, 0.0]], dtype=np.float32),
    }

    action = policy.get_action(info)

    assert action[0, 0] < 0.0
