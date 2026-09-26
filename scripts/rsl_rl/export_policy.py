import os
import glob
import argparse
import yaml
import torch
import torch.nn as nn

# --- CONFIGURACIÓN POR DEFECTO ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_BASE_DIR = os.path.join(PROJECT_ROOT, "logs", "rsl_rl")
# ---------------------------------

parser = argparse.ArgumentParser(description="Exportar a ONNX y JIT (.pt) para MLP y RNN.")
parser.add_argument("-ld", "--logdir", type=str, default=None)
parser.add_argument("-mcp", "--checkpoint", type=str, default=None)
args = parser.parse_args()


# ==============================================================================
# UTILIDADES
# ==============================================================================

def calcular_dim_observaciones(deploy_cfg):
    total_dim = 0
    obs_dict = deploy_cfg.get("observations", {})
    for obs_name, obs_data in obs_dict.items():
        history = obs_data.get("history_length", 1)
        scale_list = obs_data.get("scale", [])
        base_dim = 1 if not isinstance(scale_list, list) else len(scale_list)
        total_dim += (base_dim * history)
    return total_dim

def get_latest_log_dir(base_dir):
    all_runs = []
    for task_folder in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task_folder)
        if not os.path.isdir(task_path):
            continue
        for run_folder in os.listdir(task_path):
            run_path = os.path.join(task_path, run_folder)
            if os.path.isdir(run_path) and run_folder != "graphs":
                all_runs.append(run_path)
    if not all_runs:
        raise FileNotFoundError(f"No se encontraron entrenamientos en: {base_dir}")
    return max(all_runs, key=os.path.getmtime)

def get_latest_checkpoint(log_dir):
    checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No se encontraron checkpoints en: {log_dir}\n"
            f"Asegúrate de apuntar a la carpeta del run (ej: logs/rsl_rl/tarea/2026-XX-XX_XX-XX-XX)"
        )
    def extract_step(filepath):
        try: return int(os.path.basename(filepath).split('_')[1].split('.')[0])
        except: return -1
    return os.path.basename(max(checkpoints, key=extract_step))


# ==============================================================================
# ARQUITECTURAS STANDALONE
# Para añadir una nueva:
#   1. Define la clase con forward() y get_export_args()
#   2. Escribe su _build_xxx()
#   3. Añade la entrada en ARCHITECTURE_REGISTRY
# ==============================================================================

class StandaloneMLPPolicy(nn.Module):
    def __init__(self, num_obs, actor_hidden_dims, num_actions, activation_fn):
        super().__init__()
        layers = []
        current_dim = num_obs
        for h_dim in actor_hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation_fn)
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, num_actions))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)

    def get_export_args(self, num_obs):
        return (
            torch.randn(1, num_obs),
            ['obs'],
            ['actions'],
        )


class StandaloneRNNPolicy(nn.Module):
    def __init__(self, num_obs, rnn_hidden_dim, rnn_num_layers, actor_hidden_dims, num_actions, rnn_type, activation_fn):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=num_obs, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers, batch_first=False)
        else:
            self.rnn = nn.GRU(input_size=num_obs, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers, batch_first=False)
        layers = []
        current_dim = rnn_hidden_dim
        for h_dim in actor_hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation_fn)
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, num_actions))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs, h_in, c_in=None):
        obs_seq = obs.unsqueeze(0)
        if self.rnn_type == "lstm":
            rnn_out, (h_out, c_out) = self.rnn(obs_seq, (h_in, c_in))
            return self.actor(rnn_out.squeeze(0)), h_out, c_out
        else:
            rnn_out, h_out = self.rnn(obs_seq, h_in)
            return self.actor(rnn_out.squeeze(0)), h_out

    def get_export_args(self, num_obs):
        dummy_obs  = torch.randn(1, num_obs)
        dummy_h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        if self.rnn_type == "lstm":
            dummy_c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            return (
                (dummy_obs, dummy_h_in, dummy_c_in),
                ['obs', 'h_in', 'c_in'],
                ['actions', 'h_out', 'c_out'],
            )
        else:
            return (
                (dummy_obs, dummy_h_in),
                ['obs', 'h_in'],
                ['actions', 'h_out'],
            )


class StandaloneCustomRNNPolicy(nn.Module):
    """
    Arquitectura CustomActorCriticGo2 — estilo ANYmal (Miki et al. 2022):
      GRU recibe : obs completo (proprio + extero)
      MLP recibe : proprio directo + gru_out
    Cuando el mapa es ruidoso, el MLP aprende a ignorar el encoder
    y funciona en modo ciego con solo propiocepción.
    """
    PROPRIO_DIM = 45  # ang_vel(3)+gravity(3)+cmd(3)+joint_pos(12)+joint_vel(12)+last_action(12)

    def __init__(self, num_obs, rnn_hidden_dim, rnn_num_layers, actor_hidden_dims, num_actions, activation_fn):
        super().__init__()
        self.rnn = nn.GRU(input_size=num_obs, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers, batch_first=False)
        mlp_input_dim = self.PROPRIO_DIM + rnn_hidden_dim
        layers = []
        current_dim = mlp_input_dim
        for h_dim in actor_hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation_fn)
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, num_actions))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs, h_in):
        obs_seq = obs.unsqueeze(0)
        rnn_out, h_out = self.rnn(obs_seq, h_in)
        proprio = obs[..., :self.PROPRIO_DIM]
        mlp_in = torch.cat([proprio, rnn_out.squeeze(0)], dim=-1)
        return self.actor(mlp_in), h_out

    def get_export_args(self, num_obs):
        dummy_obs  = torch.randn(1, num_obs)
        dummy_h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        return (
            (dummy_obs, dummy_h_in),
            ['obs', 'h_in'],
            ['actions', 'h_out'],
        )

class StandaloneCustomBlindPolicy(nn.Module):
    """
    Arquitectura para el robot ciego con History Encoder y array entrelazado.
    """
    PROPRIO_DIM = 45 

    def __init__(self, num_obs, actor_hidden_dims, encoder_hidden_dims, num_actions, activation_fn):
        super().__init__()
        
        self.num_frames = int(num_obs / self.PROPRIO_DIM)
        self.num_history_obs = num_obs - self.PROPRIO_DIM
        self.latent_dim = encoder_hidden_dims[-1]
        
        # --- RECREAR ÍNDICES ENTRELAZADOS DE ISAAC LAB ---
        base_dims = [3, 3, 3, 12, 12, 12] # ang_vel, grav, cmd, pos, vel, act
        current_idx = []
        history_idx = []
        offset = 0
        
        for dim in base_dims:
            current_idx.extend(range(offset, offset + dim))
            history_idx.extend(range(offset + dim, offset + dim * self.num_frames))
            offset += dim * self.num_frames
            
        self.register_buffer("current_indices", torch.tensor(current_idx, dtype=torch.long))
        self.register_buffer("history_indices", torch.tensor(history_idx, dtype=torch.long))
        
        # 1. El Encoder
        encoder_layers = []
        curr_in = self.num_history_obs
        for h in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(curr_in, h))
            encoder_layers.append(activation_fn)
            curr_in = h
        self.history_encoder = nn.Sequential(*encoder_layers)
        
        # 2. El MLP Principal
        mlp_input_dim = self.PROPRIO_DIM + self.latent_dim
        layers = []
        curr_in = mlp_input_dim
        for h_dim in actor_hidden_dims:
            layers.append(nn.Linear(curr_in, h_dim))
            layers.append(activation_fn)
            curr_in = h_dim
        layers.append(nn.Linear(curr_in, num_actions))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        # --- SLICING NATIVO Y COMPATIBLE CON ONNX ---
        obs_current = obs[:, self.current_indices]
        obs_history = obs[:, self.history_indices]
        
        hist_latent = self.history_encoder(obs_history)
        actor_input = torch.cat([obs_current, hist_latent], dim=-1)
        
        return self.actor(actor_input)

    def get_export_args(self, num_obs):
        return (
            torch.randn(1, num_obs), # Dummy input
            ['obs'],                 # Input names
            ['actions'],             # Output names
        )
class StandaloneCustomActorCriticLidar(nn.Module):
    """Arquitectura Standalone para Sim2Real con LiDAR e Historial (Triple MLP)"""
    PROPRIO_DIM = 45 
    EXTERO_DIM = 187  

    def __init__(self, num_obs, actor_hidden_dims, encoder_hidden_dims, extero_hidden_dims, num_actions, activation_fn):
        super().__init__()
        
        self.history_len = int((num_obs - self.EXTERO_DIM) / self.PROPRIO_DIM)
        
        # Mapeamos los índices del t=0
        base_dims = [3, 3, 3, 12, 12, 12]
        current_idx = []
        offset = 0
        
        for dim in base_dims:
            current_idx.extend(range(offset, offset + dim))
            offset += dim * self.history_len
            
        self.register_buffer("current_indices", torch.tensor(current_idx, dtype=torch.long))
        
        # EL ARREGLO ESTÁ AQUÍ (Usamos el total de frames, sin restar nada)
        self.num_history_obs = self.PROPRIO_DIM * self.history_len
        self.latent_hist_dim = encoder_hidden_dims[-1]
        self.latent_extero_dim = extero_hidden_dims[-1]
        
        hist_layers = []
        curr_in = self.num_history_obs
        for h in encoder_hidden_dims:
            hist_layers.append(nn.Linear(curr_in, h))
            hist_layers.append(activation_fn)
            curr_in = h
        self.history_encoder = nn.Sequential(*hist_layers)

        extero_layers = []
        curr_in = self.EXTERO_DIM
        for h in extero_hidden_dims:
            extero_layers.append(nn.Linear(curr_in, h))
            extero_layers.append(activation_fn)
            curr_in = h
        self.extero_encoder = nn.Sequential(*extero_layers)

        self.attention_gate = nn.Sequential(
            nn.Linear(self.latent_hist_dim, self.latent_hist_dim),
            activation_fn,
            nn.Linear(self.latent_hist_dim, self.latent_extero_dim),
            nn.Sigmoid()
        )
        
        mlp_input_dim = self.PROPRIO_DIM + self.latent_hist_dim + self.latent_extero_dim
        layers = []
        curr_in = mlp_input_dim
        for h_dim in actor_hidden_dims:
            layers.append(nn.Linear(curr_in, h_dim))
            layers.append(activation_fn)
            curr_in = h_dim
        layers.append(nn.Linear(curr_in, num_actions))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        obs_current = obs[:, self.current_indices]
        obs_history = obs[:, :self.num_history_obs]
        obs_extero = obs[:, -self.EXTERO_DIM:]
        
        hist_latent = self.history_encoder(obs_history)
        extero_latent = self.extero_encoder(obs_extero)

        confidence = self.attention_gate(hist_latent)
        gated_extero = extero_latent * confidence
        
        actor_input = torch.cat([obs_current, hist_latent, gated_extero], dim=-1)
        return self.actor(actor_input)

    def get_export_args(self, num_obs):
        return (torch.randn(1, num_obs), ['obs'], ['actions'])
# ==============================================================================
# REGISTRO DE ARQUITECTURAS
# Para añadir una nueva arquitectura, añade una entrada aquí.
# ==============================================================================

def _build_mlp(policy_cfg, num_obs, num_actions, activation_fn):
    hidden_dims = policy_cfg.get("actor_hidden_dims", [512, 256, 128])
    return StandaloneMLPPolicy(num_obs, hidden_dims, num_actions, activation_fn)

def _build_rnn(policy_cfg, num_obs, num_actions, activation_fn):
    hidden_dims = policy_cfg.get("actor_hidden_dims", [512, 256, 128])
    rnn_type = policy_cfg.get("rnn_type", "gru")
    rnn_hidden = policy_cfg.get("rnn_hidden_dim", policy_cfg.get("rnn_hidden_size", 50))
    rnn_layers = policy_cfg.get("rnn_num_layers", 1)
    return StandaloneRNNPolicy(num_obs, rnn_hidden, rnn_layers, hidden_dims, num_actions, rnn_type, activation_fn)

def _build_custom_go2(policy_cfg, num_obs, num_actions, activation_fn):
    hidden_dims = policy_cfg.get("actor_hidden_dims", [512, 256, 128])
    rnn_hidden = policy_cfg.get("rnn_hidden_dim", 50)
    rnn_layers = policy_cfg.get("rnn_num_layers", 1)
    return StandaloneCustomRNNPolicy(num_obs, rnn_hidden, rnn_layers, hidden_dims, num_actions, activation_fn)

def _build_custom_blind(policy_cfg, num_obs, num_actions, activation_fn):
    actor_dims = policy_cfg.get("actor_hidden_dims", [512, 256, 128])
    enc_dims = policy_cfg.get("encoder_hidden_dims", [128, 64])
    return StandaloneCustomBlindPolicy(num_obs, actor_dims, enc_dims, num_actions, activation_fn)

def _build_custom_lidar(policy_cfg, num_obs, num_actions, activation_fn):
    actor_dims = policy_cfg.get("actor_hidden_dims", [512, 256, 128])
    enc_dims = policy_cfg.get("encoder_hidden_dims", [128, 64])
    ext_dims = policy_cfg.get("extero_hidden_dims", [128, 64])
    return StandaloneCustomActorCriticLidar(num_obs, actor_dims, enc_dims, ext_dims, num_actions, activation_fn)

ARCHITECTURE_REGISTRY = {
    "ActorCritic": _build_mlp,
    "ActorCriticRecurrent": _build_rnn,
    "CustomActorCriticGo2": _build_custom_go2,
    "CustomActorCriticBlind": _build_custom_blind,
    "CustomActorCriticLidar": _build_custom_lidar,
}


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    if args.logdir:
        if os.path.isdir(args.logdir):
            log_dir = os.path.abspath(args.logdir)

        elif os.path.isdir(os.path.join(DEFAULT_BASE_DIR, args.logdir)):
            log_dir = os.path.join(DEFAULT_BASE_DIR, args.logdir)

        elif os.sep not in args.logdir and "/" not in args.logdir:
            task_path = os.path.join(DEFAULT_BASE_DIR, args.logdir)
            if not os.path.isdir(task_path):
                raise FileNotFoundError(f"No se encontró la tarea: {task_path}")
            runs = [
                os.path.join(task_path, r)
                for r in os.listdir(task_path)
                if os.path.isdir(os.path.join(task_path, r)) and r != "graphs"
            ]
            if not runs:
                raise FileNotFoundError(f"No hay runs en: {task_path}")
            log_dir = max(runs, key=os.path.getmtime)
            print(f"[INFO] Tarea '{args.logdir}' -> run más reciente seleccionado automáticamente")
            
        else:
            raise FileNotFoundError(f"No se ha podido resolver la ruta para: {args.logdir}")
    else:
        print(f"[INFO] Buscando automáticamente el entrenamiento más reciente...")
        log_dir = get_latest_log_dir(DEFAULT_BASE_DIR)

    task_name = os.path.basename(os.path.dirname(log_dir))
    print(f"\n==================================================")
    print(f" TAREA DETECTADA: {task_name}")
    print(f"==================================================\n")
    print(f"[INFO] Usando Log Dir: {log_dir}")

    checkpoint_name = args.checkpoint if args.checkpoint else get_latest_checkpoint(log_dir)
    print(f"[INFO] Usando Checkpoint: {checkpoint_name}")

    model_path = os.path.join(log_dir, checkpoint_name)
    deploy_yaml_path = os.path.join(log_dir, "params", "deploy.yaml")
    agent_yaml_path = os.path.join(log_dir, "params", "agent.yaml")

    with open(deploy_yaml_path, 'r') as f: deploy_cfg = yaml.unsafe_load(f)
    with open(agent_yaml_path,  'r') as f: agent_cfg  = yaml.unsafe_load(f)

    num_obs = calcular_dim_observaciones(deploy_cfg)
    num_actions = len(deploy_cfg["actions"]["JointPositionAction"]["scale"]) \
        if "JointPositionAction" in deploy_cfg.get("actions", {}) else 12

    policy_cfg = agent_cfg.get("policy", {})
    class_name = policy_cfg.get("class_name", "ActorCritic")
    activation_str = policy_cfg.get("activation", "elu")
    activations = {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}
    activation_fn = activations.get(activation_str.lower(), nn.ELU())

    print(f"[INFO] Dimensiones  -> Input: {num_obs}, Output: {num_actions}")
    print(f"[INFO] Arquitectura -> {class_name}")

    # --- Instanciar arquitectura desde el registro ---
    builder = ARCHITECTURE_REGISTRY.get(class_name)
    if builder is None:
        raise ValueError(
            f"Arquitectura '{class_name}' no registrada.\n"
            f"Disponibles: {list(ARCHITECTURE_REGISTRY.keys())}"
        )
    actor_network = builder(policy_cfg, num_obs, num_actions, activation_fn)

    # --- Cargar pesos ---
    checkpoint      = torch.load(model_path, map_location='cpu', weights_only=False)
    full_state_dict = checkpoint.get('model_state_dict', checkpoint)
    actor_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith("actor."):
            actor_state_dict[key] = value
        elif key.startswith("history_encoder."):
            actor_state_dict[key] = value
        elif key.startswith("extero_encoder."): 
            actor_state_dict[key] = value
        elif key.startswith("attention_gate."):  
            actor_state_dict[key] = value
        elif key.startswith("memory_a.rnn."):
            new_key = key.replace("memory_a.", "")
            actor_state_dict[new_key] = value

    actor_network.load_state_dict(actor_state_dict, strict=False)
    actor_network.eval()

    # --- Exportar ---
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, "policy.onnx")
    pt_path   = os.path.join(export_dir, "policy.pt")

    dummy_input, input_names, output_names = actor_network.get_export_args(num_obs)

    print(f"[INFO] Exportando a ONNX (Opset 18)...")
    torch.onnx.export(
        actor_network, dummy_input, onnx_path,
        export_params=True, opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )

    traced_policy = torch.jit.trace(actor_network, dummy_input)
    traced_policy.save(pt_path)

    print(f"\n[ OK ] Archivos generados en:\n -> ONNX : {onnx_path}\n -> JIT  : {pt_path}")


if __name__ == "__main__":
    main()