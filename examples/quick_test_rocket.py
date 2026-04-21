"""
Quick test script for RocketLanding environment.

This is a minimal test to verify the environment is working correctly.
Run this first to check your installation before running the full example.
"""

if __name__ == "__main__":
    import numpy as np

    print("Testing RocketLanding environment integration...")
    print("-" * 50)

    # Test 1: Import
    print("\n[Test 1] Importing stable_worldmodel...")
    try:
        import stable_worldmodel as swm

        print("✓ Import successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        exit(1)

    # Test 2: Check PyFlyt
    print("\n[Test 2] Checking PyFlyt installation...")
    try:
        import PyFlyt.gym_envs

        print("✓ PyFlyt is installed")
    except ImportError:
        print("✗ PyFlyt is not installed")
        print("  Install with: pip install PyFlyt")
        exit(1)

    # Test 3: Create environment
    print("\n[Test 3] Creating RocketLanding environment...")
    try:
        world = swm.World(
            "swm/RocketLanding-v0",
            num_envs=2,
            image_shape=(128, 128),
            render_mode="rgb_array",
            max_episode_steps=50,  # Short episodes for quick test
        )
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # Test 4: Check spaces
    print("\n[Test 4] Checking environment spaces...")
    try:
        print(f"  Action space: {world.single_action_space.shape}")
        print(f"  Observation space: {world.single_observation_space.shape}")
        print(
            f"  Variation space: {len(world.single_variation_space.names())} variations"
        )
        print("✓ Spaces configured correctly")
    except Exception as e:
        print(f"✗ Space check failed: {e}")
        exit(1)

    # Test 5: Reset environment
    print("\n[Test 5] Resetting environment...")
    try:
        world.set_policy(swm.policy.RandomPolicy(seed=42))
        world.reset(seed=123)
        print("✓ Environment reset successful")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # Test 6: Take a few steps
    print("\n[Test 6] Taking sample steps...")
    try:
        for i in range(5):
            world.step()
        print("✓ Environment stepping works correctly")
    except Exception as e:
        print(f"✗ Stepping failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # Test 7: Check info dict
    print("\n[Test 7] Checking info dictionary...")
    try:
        world.reset(seed=456)
        world.step()
        if "pixels" in world.infos:
            print(f"  Pixels shape: {world.infos['pixels'].shape}")
        if "goal" in world.infos:
            print(f"  Goal shape: {world.infos['goal'].shape}")
        print("✓ Info dictionary contains required keys")
    except Exception as e:
        print(f"✗ Info check failed: {e}")
        exit(1)

    # Cleanup
    world.close()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    print(
        "\nYour RocketLanding environment is ready to use."
    )
    print("Run the full example with: python examples/example_rocket_landing.py")
