import os
import sys
import torch

# ================== CẤU HÌNH CƠ BẢN ==================

# Cluster muốn lấy (1-based). Ở đây là cluster số 3:
CLUSTER_ID = 3

# Tên file checkpoint chứa tất cả policies (cùng thư mục với file này)
CKPT_FILENAME = "policies_final.pth"

# Tên file output muốn lưu riêng policy cluster 3 (cùng thư mục với file này)
OUT_STATE_DICT = f"policy_{CLUSTER_ID}.pth"

# =====================================================

# Thư mục chứa file get_policy.py
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nếu muốn import myrllib (ở trên thư mục này 2 cấp: llirl_sumo)
LLIRL_SUMO_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
if LLIRL_SUMO_ROOT not in sys.path:
    sys.path.insert(0, LLIRL_SUMO_ROOT)

# 2) Import NormalMLPPolicy từ myrllib
try:
    from myrllib.policies import NormalMLPPolicy
except ImportError:
    from myrllib.policies.normal_mlp import NormalMLPPolicy


def load_policies(device):
    """
    Load LLIRL policies từ policies_final.pth (ngay cạnh script)
    - Trả về: (list[policy_model], checkpoint_dict)
    """
    policies_path = os.path.join(FILE_DIR, CKPT_FILENAME)

    if not os.path.exists(policies_path):
        raise FileNotFoundError(f"{CKPT_FILENAME} not found in {FILE_DIR}")

    print(f"[LLIRL] Loading policies from {policies_path} ...")
    ckpt = torch.load(
        policies_path,
        map_location=device,
        weights_only=False  
    )

    # Lấy thông tin kiến trúc từ checkpoint
    state_dim = ckpt.get('state_dim')
    action_dim = ckpt.get('action_dim')
    hidden_size = ckpt.get('hidden_size', 200)
    num_layers = ckpt.get('num_layers', 2)

    if state_dim is None or action_dim is None:
        raise ValueError("Checkpoint missing state_dim/action_dim in policies_final.pth")

    policy_state_dicts = ckpt.get('policies', [])
    llirl_policies = []

    hidden_sizes = (hidden_size,) * num_layers

    for i, sd in enumerate(policy_state_dicts):
        if sd is None:
            llirl_policies.append(None)
            continue

        # Tạo policy đúng kiến trúc rồi load state_dict
        policy = NormalMLPPolicy(
            state_dim,
            action_dim,
            hidden_sizes=hidden_sizes
        ).to(device)

        policy.load_state_dict(sd)
        policy.eval()
        llirl_policies.append(policy)

    print(f"[LLIRL] Loaded {len(llirl_policies)} policies (num_policies={ckpt.get('num_policies')})")
    return llirl_policies, ckpt


def main():
    device = torch.device("cpu")  # hoặc "cuda" nếu bạn muốn

    # 1) Load toàn bộ policies từ file ngay cạnh script
    llirl_policies, ckpt = load_policies(device)

    num_policies = ckpt.get("num_policies", len(llirl_policies))
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    hidden_size = ckpt["hidden_size"]
    num_layers = ckpt["num_layers"]

    print(f"[INFO] num_policies = {num_policies}")
    print(f"[INFO] state_dim={state_dim}, action_dim={action_dim}, "
          f"hidden_size={hidden_size}, num_layers={num_layers}")

    # 2) Chọn cluster cần lấy (1-based -> 0-based)
    idx = CLUSTER_ID - 1

    if idx < 0 or idx >= len(llirl_policies):
        raise IndexError(
            f"Cluster {CLUSTER_ID} không tồn tại. "
            f"len(policies) = {len(llirl_policies)}"
        )

    policy_cluster = llirl_policies[idx]
    print(f"[INFO] policy[{idx}] = {policy_cluster}")

    # 3) Lưu state_dict riêng cho cluster cần lấy (ngay cạnh script)
    out_path = os.path.join(FILE_DIR, OUT_STATE_DICT)
    torch.save(policy_cluster.state_dict(), out_path)
    print(f"[OK] Đã lưu state_dict của cluster {CLUSTER_ID} vào: {out_path}")


if __name__ == "__main__":
    main()
