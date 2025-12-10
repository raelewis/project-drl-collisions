import gymnasium as gym
import datetime, imageio, os

from stable_baselines3 import PPO

#import configs

from configs.PPO import config_0

CONFIG_NUM = 0
CONFIG = config_0
ENV_NAME = "SIMPLE"

# Eval
GIF = True
num_eval_episodes = 10
model = "MultiInputPolicy"

#MODEL_NAME = "ENV"
MODEL_NAME = f"PPO_CONFIG_{CONFIG_NUM}_{ENV_NAME}_{CONFIG.rewards}"

# Create environment
env = gym.make('drl_collisions_env:HallwayEnv-v0',
                render_mode="rgb_array",
                path=ENV_NAME, 
                episode_len=CONFIG.episode_length, 
                rewards=CONFIG.rewards)

TENSORBOARD_DIR_NAME = f"./eval/tensorboard_logs/{MODEL_NAME}_RESULTS/"
model = PPO(model,
            env,
            learning_rate=CONFIG.learning_rate, 
            batch_size=CONFIG.batch_size, 
            gamma=CONFIG.gamma, 
            gae_lambda=CONFIG.gae_lambda, 
            clip_range=CONFIG.clip_range, 
            clip_range_vf=CONFIG.clip_range_vf, 
            normalize_advantage=CONFIG.normalize_advantage, 
            ent_coef=CONFIG.ent_coef, 
            vf_coef=CONFIG.vf_coef, 
            max_grad_norm=CONFIG.max_grad_norm, 
            use_sde=CONFIG.use_sde, 
            sde_sample_freq=CONFIG.sde_sample_freq, 
            rollout_buffer_class=CONFIG.rollout_buffer_class, 
            rollout_buffer_kwargs=CONFIG.rollout_buffer_kwargs, 
            target_kl=CONFIG.target_kl, 
            stats_window_size=CONFIG.stats_window_size, 
            tensorboard_log=TENSORBOARD_DIR_NAME, 
            policy_kwargs=CONFIG.policy_kwargs, 
            verbose=CONFIG.verbose, 
            seed=CONFIG.seed, 
            device=CONFIG.device, 
            _init_setup_model=CONFIG._init_setup_model)

# Train the model
model.learn(total_timesteps=CONFIG.total_timesteps, 
            log_interval=4, 
            progress_bar=True)

# Save the model
path = os.path.join("./trained/", MODEL_NAME)
model.save(path)

# Close the environment
env.close()

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