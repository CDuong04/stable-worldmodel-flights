"""
Minimal working example for RocketLanding environment.

This is the simplest possible demonstration of the rocket landing integration.
Use this as a starting point for your own experiments.
"""

if __name__ == "__main__":
    import stable_worldmodel as swm

    # Create world
    print("Creating RocketLanding environment...")
    world = swm.World(
        "swm/RocketLanding-v0",
        num_envs=2,
        image_shape=(128, 128),
        render_mode="rgb_array",
        max_episode_steps=100,
    )

    # Set random policy
    print("Setting random policy...")
    world.set_policy(swm.policy.RandomPolicy(seed=42))

    # Reset environment
    print("Resetting environment...")
    world.reset(seed=123)

    # Take some steps
    print("Taking 10 steps...")
    for i in range(10):
        world.step()
        if i == 0:
            print(f"  Observation shape: {world.states.shape}")
            print(f"  Reward: {world.rewards}")
            print(f"  Info keys: {list(world.infos.keys())}")

    # Evaluate
    print("\nEvaluating for 3 episodes...")
    results = world.evaluate(episodes=3, seed=999)
    print(f"Success rate: {results['success_rate']:.1f}%")

    # Clean up
    world.close()
    print("\nDone! ✓")
