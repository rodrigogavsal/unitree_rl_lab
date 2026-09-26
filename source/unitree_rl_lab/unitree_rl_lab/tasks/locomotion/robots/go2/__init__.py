import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
gym.register(
    id="Unitree-Go2-Velocity-Lidar",  # <--- Este es el nuevo nombre para la terminal
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Apuntamos a tu nuevo archivo .py (asumiendo que las clases se siguen llamando RobotEnvCfg y RobotPlayEnvCfg)
        "env_cfg_entry_point": f"{__name__}.lidar_velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.lidar_velocity_env_cfg:RobotPlayEnvCfg",
        
        # El cerebro (PPO) sigue siendo el mismo, la red neuronal se adaptará sola al nuevo tamaño de entrada
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)