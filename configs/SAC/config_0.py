# Default SAC hyperparamters
learning_rate=0.0003 
buffer_size=1000000 
learning_starts=500 
batch_size=256 
tau=0.005 
gamma=0.99 
train_freq=1 
gradient_steps=1 
action_noise=None 
replay_buffer_class=None 
replay_buffer_kwargs=None 
optimize_memory_usage=False 
n_steps=1 
ent_coef='auto' 
target_update_interval=1 
target_entropy='auto' 
use_sde=False 
sde_sample_freq=-1
use_sde_at_warmup=False
stats_window_size=100
tensorboard_log=None
policy_kwargs=None
verbose=0
seed=None
device='auto'
_init_setup_model=True

# Environment Config
rewards = "dense"
total_timesteps = int(1e6)
episode_length = 500
n_episodes = total_timesteps/episode_length
eval_freq = 500