import torch
import numpy as np
from omegaconf import OmegaConf

from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk

from three_fin_geometry import ThreeFin

# 1. 載入 config
cfg = OmegaConf.load("conf/conf_thermal.yaml")
cfg.run_mode = "eval"

# 2. 建立 input keys
input_keys = [
    Key("x"), Key("y"), Key("z"),
    Key("fin_height_m"), Key("fin_height_s"),
    Key("fin_length_m"), Key("fin_length_s"),
    Key("fin_thickness_m"), Key("fin_thickness_s"),
]

# 3. 建立三個網路架構
flow_net = FullyConnectedArch(
    input_keys=input_keys, output_keys=[Key("u"), Key("v"), Key("w"), Key("p")]
)
thermal_f_net = FullyConnectedArch(input_keys=input_keys, output_keys=[Key("theta_f")])
thermal_s_net = FullyConnectedArch(input_keys=input_keys, output_keys=[Key("theta_s")])

# 4. 載入模型權重（指定路徑）
base_path = "/workspace/physicsnemo-sym/examples/three_fin_3d_mic/outputs/run_mode=eval/three_fin_thermal/"

flow_net.load_state_dict(torch.load(base_path + "flow_network.0.pth", map_location="cpu"))
thermal_f_net.load_state_dict(torch.load(base_path + "thermal_f_network.0.pth", map_location="cpu"))
thermal_s_net.load_state_dict(torch.load(base_path + "thermal_s_network.0.pth", map_location="cpu"))

flow_net.eval()
thermal_f_net.eval()
thermal_s_net.eval()

# 5. 建立幾何 sample 點（使用你的參數）
geo = ThreeFin(parameterized=True)
param = {
    "fin_height_m": 0.4,
    "fin_height_s": 0.4,
    "fin_length_m": 1.0,
    "fin_length_s": 1.0,
    "fin_thickness_m": 0.1,
    "fin_thickness_s": 0.1,
}
# 1️⃣ 流體區域推論
invar_fluid = geo.geo.sample_interior(nr_points=100000, parameterization=param)
invar_fluid_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in invar_fluid.items()}
with torch.no_grad():
    pred_flow = flow_net(invar_fluid_tensor)
    pred_theta_f = thermal_f_net(invar_fluid_tensor)

pred_fluid_np = {
    **invar_fluid,
    **{k: v.cpu().numpy() for k, v in pred_flow.items()},
    **{k: v.cpu().numpy() for k, v in pred_theta_f.items()},
}
pred_fluid_np["is_solid"] = np.zeros_like(pred_fluid_np["x"])

# ➤ 輸出 fluid_only.vtp
var_to_polyvtk(pred_fluid_np, "fluid_only.vtp")

# 2️⃣ 固體區域推論
invar_solid = geo.three_fin.sample_interior(nr_points=50000, parameterization=param)
invar_solid_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in invar_solid.items()}
with torch.no_grad():
    pred_theta_s = thermal_s_net(invar_solid_tensor)

pred_solid_np = {
    **invar_solid,
    **{k: v.cpu().numpy() for k, v in pred_theta_s.items()},
}
# 補齊空的 flow 變數，避免 paraview 出錯
for var in ["u", "v", "w", "p", "theta_f"]:
    pred_solid_np[var] = np.zeros_like(pred_solid_np["x"])
pred_solid_np["is_solid"] = np.ones_like(pred_solid_np["x"])

# ➤ 輸出 solid_only.vtp
var_to_polyvtk(pred_solid_np, "solid_only.vtp")

# 3️⃣ 合併 fluid + solid
def merge_dicts(dict1, dict2):
    return {
        k: np.concatenate([dict1.get(k, np.zeros(0)), dict2.get(k, np.zeros(0))])
        for k in set(dict1) | set(dict2)
    }

merged_result = merge_dicts(pred_fluid_np, pred_solid_np)

# ➤ 輸出 combined_fluid_solid.vtp
var_to_polyvtk(merged_result, "combined_fluid_solid.vtp")