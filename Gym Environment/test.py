from stable_baselines3 import PPO
from kartEnv import kartEnv

def main():
    # Load the trained model
    model = PPO.load("ppo_kart_final")

    # Initialize the environment
    env = kartEnv()

    # Reset the environment
    obs, _ = env.reset()

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Render the environment to visualize performance
        env.render()

        # Get the action from the trained model
        action, _states = model.predict(obs, deterministic=True)

        # Step the environment with the selected action
        obs, reward, terminated, truncated, _ = env.step(action)

        # Accumulate the total reward
        total_reward += reward

    print(f"Total reward: {total_reward}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()