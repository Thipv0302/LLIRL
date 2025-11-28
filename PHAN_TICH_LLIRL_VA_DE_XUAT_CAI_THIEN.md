# PHÂN TÍCH LLIRL VÀ ĐỀ XUẤT CẢI THIỆN

## 1. PHÂN TÍCH TÌNH TRẠNG HIỆN TẠI

### 1.1. Kết quả So sánh
- **DDQN**: -68,981 (tốt nhất) ✅
- **LLIRL**: -147,864 (kém hơn DDQN 2.14 lần) 
- **PPO**: -196,014 (kém nhất)

### 1.2. Vấn đề với LLIRL

#### A. Clustering không hoạt động đúng
```
- Tất cả 5 periods → cluster 1
- num_clusters = 1 (sau 5 periods)
- cluster_assignments: [1, 1, 1, 1, 1]
```
**Vấn đề**: LLIRL không phân biệt được các môi trường khác nhau, tất cả đều được gán vào 1 cluster.

#### B. Policy Selection không đa dạng
```
- Tất cả periods chọn "cluster" method
- Luôn chọn policy 1
- Không có performance-based selection
```

#### C. Performance giảm dần theo thời gian
```
Period 2: -126,967 (tốt nhất)
Period 3: -305,454
Period 4: -473,339
Period 5: -844,526 (tệ nhất)
```
**Vấn đề**: Model không học tốt, performance giảm dần thay vì cải thiện.

#### D. Episode Length không tối ưu
```
- LLIRL: 360 steps/episode
- DDQN: 103 steps/episode
```
**Vấn đề**: LLIRL chạy lâu hơn nhưng reward thấp hơn → không hiệu quả.

---

## 2. NGUYÊN NHÂN

### 2.1. Clustering Parameters không phù hợp
- **zeta = 1.0**: Quá thấp, khó tạo cluster mới
- **sigma = 0.25**: Có thể quá cao, làm mất độ nhạy
- **tau1, tau2 = 1.0**: Temperature parameters có thể cần điều chỉnh
- **EM_STEPS = 1**: Quá ít, cần nhiều iterations hơn để convergence

### 2.2. Policy Training Parameters
- **Learning rate = 0.003**: Có thể quá cao, gây instability
- **num_iter = 50**: Có thể không đủ cho convergence
- **REINFORCE**: Algorithm đơn giản, variance cao
- **Không có baseline**: Tăng variance trong gradient estimation

### 2.3. Environment Clustering Logic
- Các task parameters khác nhau nhưng đều được gán vào cluster 1
- Có thể do:
  - Environment models không đủ khác biệt
  - Likelihood computation không chính xác
  - Prior distribution quá mạnh về cluster 1

---

## 3. ĐỀ XUẤT CẢI THIỆN

### 3.1. Cải thiện Clustering

#### A. Điều chỉnh CRP Parameters
```python
# Trong env_clustering.py
ZETA = 0.5  # Giảm từ 1.0 → 0.5 (dễ tạo cluster mới hơn)
SIGMA = 0.1  # Giảm từ 0.25 → 0.1 (tăng độ nhạy)
TAU1 = 0.5  # Giảm temperature cho likelihood
TAU2 = 0.5  # Giảm temperature cho prior
EM_STEPS = 5  # Tăng từ 1 → 5 (nhiều iterations hơn)
```

#### B. Cải thiện Environment Model Architecture
```python
# Tăng capacity của env models
env_hidden_size = 256  # Tăng từ 200 → 256
env_num_layers = 3     # Tăng từ 2 → 3
```

#### C. Thêm Regularization
```python
# Thêm L2 regularization cho env models
weight_decay = 1e-4
```

### 3.2. Cải thiện Policy Training

#### A. Sử dụng PPO thay vì REINFORCE
```python
# Trong policy_training.py
--algorithm ppo  # Thay vì reinforce
--clip 0.2
--epochs 5
--baseline linear
```

**Lý do**: 
- PPO ổn định hơn REINFORCE
- Có clipping để tránh policy update quá lớn
- Có baseline để giảm variance

#### B. Điều chỉnh Learning Rate
```python
--lr 1e-3  # Giảm từ 0.003 → 0.001
--lr_decay 0.95  # Learning rate decay
--lr_min 1e-5  # Minimum learning rate
```

#### C. Tăng số iterations
```python
--num_iter 100  # Tăng từ 50 → 100
```

#### D. Thêm Baseline
```python
--use_baseline  # Enable baseline
--baseline linear  # Linear feature baseline
```

#### E. Gradient Clipping
```python
--grad_clip 0.5  # Clip gradients
```

### 3.3. Cải thiện Policy Selection

#### A. Tăng weight cho Performance-based
```python
--policy_eval_weight 0.7  # Tăng từ 0.5 → 0.7
--num_test_episodes 5  # Tăng từ 3 → 5
```

#### B. Sử dụng General Policy
```python
--use_general_policy  # Đã có, nhưng cần đảm bảo hoạt động đúng
```

### 3.4. Cải thiện Evaluation

#### A. Tăng số episodes để đánh giá
```python
--num_episodes 20  # Tăng từ 10 → 20
```

#### B. Sử dụng multiple seeds
```python
--seeds 42 100 200  # Test với nhiều seeds
```

---

## 4. KẾ HOẠCH THỰC HIỆN

### Bước 1: Cải thiện Clustering
```bash
cd llirl_sumo
python env_clustering.py \
    --sumo_config ../nets/120p4k/run_120p4k.sumocfg \
    --model_path saves/120p4k_v2 \
    --et_length 1 \
    --num_periods 5 \
    --device cuda \
    --seed 1009 \
    --batch_size 8 \
    --env_num_layers 3 \
    --env_hidden_size 256 \
    --H 4 \
    --max_steps 7200
```

**Thay đổi trong code**:
- `ZETA = 0.5` (trong env_clustering.py)
- `SIGMA = 0.1`
- `TAU1 = 0.5, TAU2 = 0.5`
- `EM_STEPS = 5`

### Bước 2: Cải thiện Policy Training
```bash
cd llirl_sumo
python policy_training.py \
    --sumo_config ../nets/120p4k/run_120p4k.sumocfg \
    --model_path saves/120p4k_v2 \
    --output output/120p4k_v2 \
    --algorithm ppo \
    --opt adam \
    --lr 1e-3 \
    --num_iter 100 \
    --num_periods 5 \
    --device cuda \
    --seed 1009 \
    --batch_size 8 \
    --hidden_size 200 \
    --num_layers 2 \
    --use_general_policy \
    --num_test_episodes 5 \
    --policy_eval_weight 0.7 \
    --max_steps 7200 \
    --use_baseline \
    --baseline linear \
    --clip 0.2 \
    --epochs 5 \
    --grad_clip 0.5 \
    --lr_decay 0.95 \
    --lr_min 1e-5
```

### Bước 3: So sánh lại
```bash
python compare_3_models_120p4k.py \
    --sumo_config nets/120p4k/run_120p4k.sumocfg \
    --ddqn_model_path ddqn_sumo/output/120p4k/ddqn_model_final.pth \
    --llirl_model_path llirl_sumo/saves/120p4k_v2 \
    --ppo_model_path ppo_sumo/saves/120p4k/ppo_policy_final.pth \
    --output output/comparison_120p4k_v2 \
    --num_episodes 20 \
    --llirl_policy_selection best
```

---

## 5. CÁC THAY ĐỔI CODE CẦN THIẾT

### 5.1. env_clustering.py
```python
# Dòng ~30-40
ZETA = 0.5  # Giảm từ 1.0
SIGMA = 0.1  # Giảm từ 0.25
TAU1 = 0.5  # Giảm từ 1.0
TAU2 = 0.5  # Giảm từ 1.0
EM_STEPS = 5  # Tăng từ 1
```

### 5.2. policy_training.py
- Đảm bảo PPO được sử dụng thay vì REINFORCE
- Thêm baseline nếu chưa có
- Cải thiện learning rate scheduling




