# TRAFFIC-LLIRL (DDQN Version)

This project implements the **Lifelong Incremental Reinforcement Learning (LLIRL)** algorithm combined with a **Double Deep Q-Network (DDQN)** policy to control traffic light timings in the **SUMO (Simulation of Urban MObility)** environment.

The main goal of this project is to construct a machine learning-based traffic control system capable of long-term adaptive learning across various traffic scenarios. By clustering different environment scenarios into specific tasks, the system trains and applies online learning to each task using the DDQN algorithm.

## 📂 Directory Structure

- **`baseline/`**: Contains traditional baseline scripts (e.g., Fixed-time, Actuated, Max-Pressure, etc.) for performance comparison against the machine learning model.
- **`ddqn_sumo/`**: The core directory housing scripts for training and applying standard DDQN on SUMO, without the LLIRL mechanism.
- **`llirl_sumo/`**: The main core directory implementing the environment clustering system and task training using the LLIRL algorithm paired with a DDQN policy.
- **`nets/`**: Contains SUMO traffic network configurations (e.g., `60p2k`, `demo`). These include lane definitions (`.net.xml`), traffic routes/flows (`.rou.xml`), and the main configuration files (`.sumocfg`).
- **Data / Output directories**:
  - `report_random/`, `result_random/`, `result_test/`: Store logs, `summary.xml` files, performance charts, and statistics from model applications or baselines.
- **Main execution scripts** are located in the root directory for quickly launching workflows (see the "Usage" section for details).

## 🛠 Prerequisites

Ensure you have the following tools installed on your system:
- **Python 3.8+**
- **SUMO (Simulation of Urban MObility)** (A compatible version with your code; ensure the `SUMO_HOME` environment variable is properly configured).

Install the required Python libraries using pip:
```bash
pip install -r requirements.txt
```

Key library dependencies include:
- `torch >= 2.0.0`
- `gym >= 0.15.0`
- `numpy >= 1.16.0, < 2.0.0`
- `scipy >= 1.3.0`
- `tqdm >= 4.40.0`, `matplotlib >= 3.0.0`

---

## 🚀 Usage

The project execution is designed for multiple modes (Baseline, DDQN Standalone, LLIRL Train, LLIRL Apply). The commands below are triggered by running the Python scripts directly in the root directory.

### 1. Training with the LLIRL System

The **`train_llirl_ddqn.py`** file handles the complete LLIRL training lifecycle using DDQN. This algorithm runs two automatic steps:
- **Step 1 — Environment Clustering**: Classifies and extracts features from the set of environment configurations in the `nets/60p2k` directory. The script calls the `env_clustering.py` system located in `llirl_sumo/`.
- **Step 2 — Policy Training**: Trains the DDQN (Double Deep Q-Network) for each task based on the analyzed clusters. This process calls `policy_ddqn_training.py` inside `llirl_sumo/`.
  
Execution command:
```bash
python train_llirl_ddqn.py
```
> **Note**: You can edit `train_llirl_ddqn.py` to change the `sumo_config` variable to point to your desired network folder, or adjust hyperparameters such as `lr`, `batch_size`, and `episodes`.

### 2. Applying the LLIRL Model (Apply / Eval)

The **`apply_llirl.py`** file uses the trained LLIRL model to test on a new traffic environment or specifically designated periods. Similar to the training process, this script follows 2 steps:
- Runs `env_clustering_apply.py` (Only needs to be run once per new environment).
- Runs `policy_ddqn_apply.py` to perform **online learning** directly on the actual target environment.

Execution command:
```bash
python apply_llirl.py
```
> XML results and metrics will be saved into `.result_test/h/` (or your configured output path).

### 3. Standard DDQN & Baseline Methods

- **Training DDQN Standalone**: 
  ```bash
  python train_ddqn.py
  ```
  Trains a standard DDQN model, bypassing the LLIRL logic. This script directly references `ddqn_sumo/ddqn_training.py`.

- **Applying DDQN Standalone**:
  ```bash
  python apply_ddqn.py
  ```

- **Applying Baselines**:
  ```bash
  python apply_baseline.py
  ```
  Use this to test and compare results using fixed-time or actuated algorithms.

---

## 📈 Additional Utilities

#### Grouping and Averaging Summaries (`export_summary_avg.py`)

Since the SUMO simulation generates numerous `summary_ep_*.xml` files per episode batch, the **`export_summary_avg.py`** script allows you to automatically parse multiple XML files in a specific directory. It then calculates the average values for corresponding `step time` intervals and outputs a single `summary_avg.xml` file.

```bash
python export_summary_avg.py
```
> Modify the `FOLDER` constant inside this file depending on where you want to calculate the data averages. This is typically used to plot more stable and generalized charts.

---

## ⚙️ Hyperparameter Configuration (For Research)
- Learning parameters like `lr`, `gamma`, `epsilon_decay` or the Neural Network structure `[200, 200]` can be fine-tuned by passing flag arguments into the `subprocess` commands within the root `train_...` and `apply_...` files. The codebase structure is modular, and all logging and model saving setups are centralized for easy monitoring.
