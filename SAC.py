import gymnasium as gym
import imageio, os

from stable_baselines3 import HerReplayBuffer, SAC

#import configs
from configs.SAC import config_0

CONFIG_NUM = 0
CONFIG = config_0              
ENV_NAME = "CORNER_OBS"         # Which eval should be utilized?
HER = True                      # Should SAC utilize Hindsight Experience Replay?

# Eval
GIF = True                      # Create gif for each num_eval_episodes         
SAVE = False                    # Save model and replay buffer of model?
num_eval_episodes = 10          # Number of episodes used in evaluation
model = "MultiInputPolicy"

MODEL_NAME = f"SAC_CONFIG_{CONFIG_NUM}_{ENV_NAME}_{CONFIG.rewards}"

# Create environment
env = gym.make('drl_collisions_env:HallwayEnv-v0',
                render_mode="rgb_array",
                path=ENV_NAME, 
                episode_len=CONFIG.episode_length, 
                rewards=CONFIG.rewards)


if HER:
    if CONFIG.rewards != "sparse":
        MODEL_NAME += "_HER"
        TENSORBOARD_DIR_NAME = f"./eval/tensorboard_logs/{MODEL_NAME}_RESULTS/"
        model = SAC(model,
                    env,
                    learning_rate=CONFIG.learning_rate, 
                    buffer_size=CONFIG.buffer_size, 
                    learning_starts=CONFIG.learning_starts, 
                    batch_size=CONFIG.batch_size, 
                    tau=CONFIG.tau, 
                    gamma=CONFIG.gamma,
                    train_freq=CONFIG.train_freq, 
                    gradient_steps=CONFIG.gradient_steps, 
                    action_noise=CONFIG.action_noise, 
                    optimize_memory_usage=CONFIG.optimize_memory_usage, 
                    ent_coef=CONFIG.ent_coef, 
                    target_update_interval=CONFIG.target_update_interval, 
                    target_entropy=CONFIG.target_entropy, 
                    use_sde=CONFIG.use_sde, 
                    sde_sample_freq=CONFIG.sde_sample_freq,
                    use_sde_at_warmup=CONFIG.use_sde_at_warmup,
                    stats_window_size=CONFIG.stats_window_size,
                    policy_kwargs=CONFIG.policy_kwargs,
                    verbose=CONFIG.verbose,
                    seed=CONFIG.seed,
                    device=CONFIG.device,
                    _init_setup_model=CONFIG._init_setup_model,
                    tensorboard_log=TENSORBOARD_DIR_NAME,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=CONFIG.replay_buffer_kwargs)
    else:
        print("Cannot utilize HER with sparse rewards. Please change reward type to dense or set HER to False.")
        exit()
else:
    TENSORBOARD_DIR_NAME = f"./eval/tensorboard_logs/{MODEL_NAME}_RESULTS/"
    model = SAC(model,
                env,
                learning_rate=CONFIG.learning_rate, 
                buffer_size=CONFIG.buffer_size, 
                learning_starts=CONFIG.learning_starts, 
                batch_size=CONFIG.batch_size, 
                tau=CONFIG.tau, 
                gamma=CONFIG.gamma,
                train_freq=CONFIG.train_freq, 
                gradient_steps=CONFIG.gradient_steps, 
                action_noise=CONFIG.action_noise, 
                optimize_memory_usage=CONFIG.optimize_memory_usage, 
                ent_coef=CONFIG.ent_coef, 
                target_update_interval=CONFIG.target_update_interval, 
                target_entropy=CONFIG.target_entropy, 
                use_sde=CONFIG.use_sde, 
                sde_sample_freq=CONFIG.sde_sample_freq,
                use_sde_at_warmup=CONFIG.use_sde_at_warmup,
                stats_window_size=CONFIG.stats_window_size,
                policy_kwargs=CONFIG.policy_kwargs,
                verbose=CONFIG.verbose,
                seed=CONFIG.seed,
                device=CONFIG.device,
                _init_setup_model=CONFIG._init_setup_model,
                tensorboard_log=TENSORBOARD_DIR_NAME)

# Train the agent
model.learn(total_timesteps=CONFIG.total_timesteps, 
            log_interval=4, 
            progress_bar=True)

# Save the model
if SAVE:
    path = os.path.join("./trained/", MODEL_NAME)
    path_replay = os.path.join("./trained/", f"{MODEL_NAME}_REPLAY")
    model.save(path)
    model.save_replay_buffer(path_replay)

# Close the environment
env.close()

# # Create a GIF of the trained agent in action
if GIF:
    # Regenerate the env
    env = gym.make('drl_collisions_env:HallwayEnv-v0', 
               render_mode="rgb_array", 
               path=ENV_NAME, 
               episode_len=CONFIG.episode_length, 
               rewards=CONFIG.rewards)
    
    images = []
    obs = model.env.reset()
    img = model.env.render(mode="rgb_array")
    for i in range(CONFIG.episode_length):
        images.append(img)
        action, _ = model.predict(obs)
        obs, rewards, done, info = model.env.step(action)
        img = model.env.render(mode="rgb_array")
        images.append(img)
        if done:
            break # Reset if episode ends
    env.close()
    imageio.mimsave(f"./eval/gifs/{MODEL_NAME}.gif", images, fps=30)

# Delete the model to save memory
del model