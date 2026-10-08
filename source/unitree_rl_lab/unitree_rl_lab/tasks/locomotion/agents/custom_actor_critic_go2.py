from __future__ import annotations
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import MLP
from rsl_rl.utils import unpad_trajectories  


class CustomActorCriticRNNLidar(ActorCriticRecurrent):
    PROPRIO_DIM = 45

    def __init__(self, obs, obs_groups, num_actions, rnn_hidden_dim=50, **kwargs):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            rnn_hidden_dim=rnn_hidden_dim,
            **kwargs,
        )
        mlp_input_dim = self.PROPRIO_DIM + rnn_hidden_dim  # 45 + 50 = 95
        actor_hidden_dims = kwargs.get("actor_hidden_dims", [512, 256, 128])
        activation = kwargs.get("activation", "elu")
        self.actor = MLP(mlp_input_dim, num_actions, actor_hidden_dims, activation)

        print(f"[CustomActorCriticRNNLidar] GRU input: 232 | MLP input: {mlp_input_dim} "
              f"(proprio={self.PROPRIO_DIM} + gru={rnn_hidden_dim})")

    def act(self, obs, masks=None, hidden_states=None):
        obs_tensor = self.get_actor_obs(obs)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        proprio = obs_tensor[..., :self.PROPRIO_DIM]

        gru_out = self.memory_a(obs_tensor, masks, hidden_states).squeeze(0)

        # Durante PPO update, memory_a hace unpad internamente — hay que hacer lo mismo a proprio
        if masks is not None:
            proprio = unpad_trajectories(proprio, masks)

        mlp_in = torch.cat([proprio, gru_out], dim=-1)
        self.update_distribution(mlp_in)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs_tensor = self.get_actor_obs(obs)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        proprio = obs_tensor[..., :self.PROPRIO_DIM]
        gru_out = self.memory_a(obs_tensor).squeeze(0)
        mlp_in = torch.cat([proprio, gru_out], dim=-1)
        return self.actor(mlp_in)



class CustomActorCriticBlind(ActorCritic):
    PROPRIO_DIM = 45 

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        actor_hidden_dims = kwargs.get("actor_hidden_dims", [512, 256, 128])
        activation = kwargs.get("activation", "elu")
        encoder_hidden_dims = kwargs.get("encoder_hidden_dims", [128, 64])

        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)

        total_actor_obs = next(self.actor.parameters()).shape[1]
        # Total frames = (Total obs) / (Propiocepción)
        # Ej: 495 / 45 = 11 frames (1 presente + 10 historial)
        self.num_frames = int(total_actor_obs / self.PROPRIO_DIM)
        
        # 2. Mapear los índices del Presente (t=0) y del Historial (t < 0)
        base_dims = [3, 3, 3, 12, 12, 12] # ang_vel, grav, cmd, pos, vel, act
        current_idx = []
        history_idx = []
        offset = 0
        
        for dim in base_dims:
            # El presente (t=0) son los primeros 'dim' valores del bloque de cada sensor
            current_idx.extend(range(offset, offset + dim))
            # El historial son los valores restantes de ese bloque
            history_idx.extend(range(offset + dim, offset + dim * self.num_frames))
            offset += dim * self.num_frames
            
        # Registramos ambos en buffers para que se exporten dentro del ONNX
        self.register_buffer("current_indices", torch.tensor(current_idx, dtype=torch.long))
        self.register_buffer("history_indices", torch.tensor(history_idx, dtype=torch.long))
        
        # 3. Calcular dimensiones
        self.num_history_obs = self.PROPRIO_DIM * (self.num_frames - 1)
        self.latent_dim = encoder_hidden_dims[-1] 
        
        # 4. Construir el Encoder del Historial dinámicamente
        encoder_layers = []
        curr_in = self.num_history_obs
        for h in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(curr_in, h))
            encoder_layers.append(nn.ELU()) 
            curr_in = h
        self.history_encoder = nn.Sequential(*encoder_layers)
        
        # 5. Reconstruir el Actor principal
        mlp_input_dim = self.PROPRIO_DIM + self.latent_dim
        self.actor = MLP(mlp_input_dim, num_actions, actor_hidden_dims, activation)
        
        print(
            f"[CustomActorCriticBlind] Sim2Real Automático | "
            f"Frames totales detectados: {self.num_frames} | "
            f"Historial a procesar: {self.num_history_obs} -> {encoder_hidden_dims} | "
            f"Entrada final al MLP: {mlp_input_dim}"
        )

    def act(self, obs, **kwargs):
        obs_tensor = self.get_actor_obs(obs)
        if hasattr(self, 'actor_obs_normalizer'):
            obs_tensor = self.actor_obs_normalizer(obs_tensor)
            
        # --- SLICING MATEMÁTICO PURO (100% C++ y ONNX compatible) ---
        obs_current = obs_tensor[:, self.current_indices]
        obs_history = obs_tensor[:, self.history_indices]
        
        hist_latent = self.history_encoder(obs_history)
        actor_input = torch.cat([obs_current, hist_latent], dim=-1)
        
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs_tensor = self.get_actor_obs(obs)
        if hasattr(self, 'actor_obs_normalizer'):
            obs_tensor = self.actor_obs_normalizer(obs_tensor)
            
        obs_current = obs_tensor[:, self.current_indices]
        obs_history = obs_tensor[:, self.history_indices]
        
        hist_latent = self.history_encoder(obs_history)
        actor_input = torch.cat([obs_current, hist_latent], dim=-1)
        
        return self.actor(actor_input)



class CustomActorCriticLidar(ActorCritic):
    PROPRIO_DIM = 45 
    EXTERO_DIM = 187  

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        actor_hidden_dims = kwargs.get("actor_hidden_dims", [512, 256, 128])
        activation = kwargs.get("activation", "elu")
        encoder_hidden_dims = kwargs.get("encoder_hidden_dims", [128, 64])
        extero_hidden_dims = kwargs.get("extero_hidden_dims", [128, 64]) 

        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)

        total_actor_obs = next(self.actor.parameters()).shape[1]
        
        self.history_len = int((total_actor_obs - self.EXTERO_DIM) / self.PROPRIO_DIM)
        
        # 1. Mapear los índices de t=0 automáticamente (Sim2Real puro)
        base_dims = [3, 3, 3, 12, 12, 12]
        current_idx = []
        offset = 0
        for dim in base_dims:
            current_idx.extend(range(offset, offset + dim))
            offset += dim * self.history_len  
            
        self.register_buffer("current_indices", torch.tensor(current_idx, dtype=torch.long))

        # 2. Matemáticas de dimensiones
        self.num_history_obs = self.PROPRIO_DIM * self.history_len 
        self.latent_hist_dim = encoder_hidden_dims[-1]
        self.latent_extero_dim = extero_hidden_dims[-1]
        
        # 3. Construir Encoder del Historial 
        hist_layers = []
        curr_in = self.num_history_obs
        for h in encoder_hidden_dims:
            hist_layers.append(nn.Linear(curr_in, h))
            hist_layers.append(nn.ELU())
            curr_in = h
        self.history_encoder = nn.Sequential(*hist_layers)

        # 4. Construir Encoder Exteroceptivo (LiDAR)
        extero_layers = []
        curr_in = self.EXTERO_DIM
        for h in extero_hidden_dims:
            extero_layers.append(nn.Linear(curr_in, h))
            extero_layers.append(nn.ELU())
            curr_in = h
        self.extero_encoder = nn.Sequential(*extero_layers)
    
        # Extrae la confianza (0.0 a 1.0) basándose en la propiocepción
        self.attention_gate = nn.Sequential(
            nn.Linear(self.latent_hist_dim, self.latent_hist_dim),
            nn.ELU(),
            nn.Linear(self.latent_hist_dim, self.latent_extero_dim),
            nn.Sigmoid() # Fuerza la salida al rango [0, 1]
        )
        
        # 5. Reconstruir el Actor principal
        mlp_input_dim = self.PROPRIO_DIM + self.latent_hist_dim + self.latent_extero_dim
        self.actor = MLP(mlp_input_dim, num_actions, actor_hidden_dims, activation)
        
        print(
            f"[CustomActorCriticLidar - Gated] Listo | "
            f"Frames: {self.history_len} | Input: {total_actor_obs} dims."
        )

    def act(self, obs, **kwargs):
        obs_tensor = self.get_actor_obs(obs)
        if hasattr(self, 'actor_obs_normalizer'):
            obs_tensor = self.actor_obs_normalizer(obs_tensor)

        # A. Slicing Matemático
        obs_current = obs_tensor[:, self.current_indices]
        obs_history = obs_tensor[:, :self.num_history_obs]
        obs_extero = obs_tensor[:, -self.EXTERO_DIM:]
        
        # B. Encoders Independientes
        hist_latent = self.history_encoder(obs_history)
        extero_latent = self.extero_encoder(obs_extero)
        
        # C. Aplicar la Puerta Sigmoide
        confidence = self.attention_gate(hist_latent)
        gated_extero = extero_latent * confidence  # Multiplicación elemento a elemento
        
        # D. Concatenar y Actuar
        actor_input = torch.cat([obs_current, hist_latent, gated_extero], dim=-1)
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs_tensor = self.get_actor_obs(obs)
        if hasattr(self, 'actor_obs_normalizer'):
            obs_tensor = self.actor_obs_normalizer(obs_tensor)
            
        obs_current = obs_tensor[:, self.current_indices]
        obs_history = obs_tensor[:, :self.num_history_obs]
        obs_extero = obs_tensor[:, -self.EXTERO_DIM:]
        
        hist_latent = self.history_encoder(obs_history)
        extero_latent = self.extero_encoder(obs_extero)
        
        # Aplicar la Puerta Sigmoide también en inferencia
        confidence = self.attention_gate(hist_latent)
        gated_extero = extero_latent * confidence
        
        actor_input = torch.cat([obs_current, hist_latent, gated_extero], dim=-1)
        return self.actor(actor_input)