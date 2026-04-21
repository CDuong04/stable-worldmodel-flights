"""
Example script demonstrating the RocketLanding environment integration.

This script shows how to:
1. Create a World with the RocketLanding environment
2. Inspect available variations
3. Collect training data with a random policy
4. Record videos (optional)

This is a basic proof of concept. Training world models and implementing
expert policies will be done in later stages.
"""

if __name__ == "__main__":
    import stable_worldmodel as swm

    print("=" * 70)
    print("Rocket Landing Environment - Proof of Concept")
    print("=" * 70)

    ######################
    ##  World Creation  ##
    ######################

    print("\n[1/4] Creating World with RocketLanding environment...")

    try:
        world = swm.World(
            "swm/RocketLanding-v0",
            num_envs=4,  # Use 4 parallel environments
            image_shape=(224, 224),  # Resize images to 224x224
            render_mode="rgb_array",
            max_episode_steps=500,  # Rocket landing episodes can be longer
        )
        print("✓ World created successfully!")
    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nTo use the RocketLanding environment, you need to install PyFlyt:")
        print("  pip install PyFlyt")
        print("\nFor GPU-accelerated rendering (optional):")
        print("  pip install PyFlyt[viz]")
        exit(1)

    #######################
    ##  Inspect Spaces   ##
    #######################

    print("\n[2/4] Inspecting environment spaces...")

    print("\n  Action Space:")
    print(f"    {world.single_action_space}")
    print("    Actions control: finlets (x, y, roll), booster (ignition, throttle, gimbal x, y)")

    print("\n  Observation Space:")
    print(f"    {world.single_observation_space}")

    print("\n  Available Variations:")
    variations = world.single_variation_space.names()
    for var in variations:
        print(f"    - {var}")

    print("\n  Variation Details:")
    print("    Rocket variations:")
    print("      • start_pos: Initial position (x, y, z) in meters")
    print("      • start_vel: Initial velocity (vx, vy, vz) in m/s")
    print("      • start_orn: Initial orientation (roll, pitch, yaw) in radians")
    print("      • starting_fuel_ratio: Fuel level (0.01 to 0.2)")
    print("      • start_ang_vel: Initial angular velocity (rad/s)")
    print("    Landing pad variations:")
    print("      • position: Landing pad position (x, y) in meters")

    #######################
    ##  Data Collection  ##
    #######################

    print("\n[3/4] Collecting training data with random policy...")
    print("  (This may take a few minutes depending on episode length)")

    world.set_policy(swm.policy.RandomPolicy(seed=42))

    # Collect a small dataset for proof of concept
    # For full training, you'd want hundreds or thousands of episodes
    world.record_dataset(
        "rocket-landing-demo",
        episodes=10,  # Small number for quick testing
        seed=2347,
        options={
            "variation": (
                "rocket.start_pos",
                "rocket.start_vel",
                "rocket.start_orn",
            )
        },
    )

    print("✓ Dataset collected successfully!")
    print("  Dataset saved to: ~/.cache/stable_worldmodel/rocket-landing-demo")

    #####################
    ##  Video Recording ##
    #####################

    print("\n[4/4] Recording a sample video...")
    print("  (Optional - comment this out if you want faster execution)")

    # Uncomment to record a video of the rocket landing attempt
    # Note: This will create video files in the current directory
    try:
        import os

        os.makedirs("./rocket_videos", exist_ok=True)
        world.record_video(
            "./rocket_videos",
            max_steps=500,
            fps=30,
            seed=9999,
            options={
                "variation": (
                    "rocket.start_pos",
                    "rocket.start_vel",
                )
            },
        )
        print("✓ Video saved to: ./rocket_videos/")
    except Exception as e:
        print(f"  (Video recording skipped: {e})")

    ######################
    ##  Evaluation Test ##
    ######################

    print("\n[5/5] Testing evaluation with random policy...")

    # Test the evaluation function
    world.set_policy(swm.policy.RandomPolicy(seed=123))
    results = world.evaluate(
        episodes=5,  # Quick test with 5 episodes
        seed=5555,
        options={
            "variation": (
                "rocket.start_pos",
                "rocket.start_vel",
            )
        },
    )

    print(f"\n  Evaluation Results:")
    print(f"    Success Rate: {results['success_rate']:.1f}%")
    print(f"    Episode Successes: {results['episode_successes']}")

    # Clean up
    world.close()

    print("\n" + "=" * 70)
    print("Proof of Concept Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Collect more data with various initial conditions")
    print("  2. Train a world model (e.g., DINOWM) on the collected data")
    print("  3. Implement an expert policy for better data collection")
    print("  4. Use the trained world model with a solver (CEM, GD, MPPI)")
    print("  5. Evaluate the world model policy's performance")
    print("\nFor world model training, see: scripts/train/dinowm.py")
    print("For policy examples, see: stable_worldmodel/policy.py")
    print("=" * 70)
