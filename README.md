# Project-DRL-Collisions
Author: Rachel Lewis

This project contains the source code for my Master's Project: 'Can  Deep  Reinforcement  Learning  Help  Robots  Avoid  Unexpected  Obstacles?'

## Installing dependencies
First, create a virtual environment of your choice. If using pip, you can run the following to install all dependencies:
`pip install -r requirements.txt`


## Running the Code
When running the environments, edit or create new config files for SAC[https://stable-baselines3.readthedocs.io/en/master/modules/sac.html] and [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), following the naming conventions in the config_0.py files.

Simulation environment is a custom [Gymnasium](https://gymnasium.farama.org/) environment, inspired by the [D4RL](https://arxiv.org/abs/2004.07219) environments. 

## Sources

D4RL
```
@misc{fu2021d4rldatasetsdeepdatadriven,
      title={D4RL: Datasets for Deep Data-Driven Reinforcement Learning}, 
      author={Justin Fu and Aviral Kumar and Ofir Nachum and George Tucker and Sergey Levine},
      year={2021},
      eprint={2004.07219},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2004.07219}, 
}
```

Stable-Baselines3
```
@article{JMLR:v22:20-1364,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1--8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

Gymnasium Robotics
```@software{gymnasium_robotics2023github,
  author = {Rodrigo de Lazcano and Kallinteris Andreas and Jun Jet Tai and Seungjae Ryan Lee and Jordan Terry},
  title = {Gymnasium Robotics},
  url = {http://github.com/Farama-Foundation/Gymnasium-Robotics},
  version = {1.3.1},
  year = {2024},
}```

MuJoCo
```@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}```