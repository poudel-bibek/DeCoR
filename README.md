---
### 🎯  Reproducing the Results
The details on setup and testing are given below.

####  ⚙️ Setup

- Install [SUMO version 1.21](https://github.com/eclipse-sumo/sumo/releases/tag/v1_21_0)  by following [official instructions](https://github.com/eclipse-sumo/sumo).
- To verify SUMO is installed and accessible, open terminal and enter:
  ```bash
  sumo --version
  ```
- Install Python version 3.12 ([Anaconda 2024.06](https://repo.anaconda.com/archive/) recommended)
- Create and activate a virtual environment (recommended):
    ```bash
     conda create -n decor python=3.12 -y
     conda activate decor
     ```

- Go to the code folder and use the following command to install required packages
    ```bash
     pip install -r requirements.txt
     ```

#### 🚀 Training

__Step 1:__ Complete the setup

__Step 2:__ Make sure config.py file has `evaluate` set to `False`  (set `gui` set to `False` for faster training)

__Step 3:__ Run the following command
```bash
python main.py
```
__Step 4:__  To monitor progress using TensorBoard, open another terminal, go to the code folder and enter the following command:
```bash
tensorboard --logdir=./runs
```
__Step 5:__  The folder corresponding to the training run will be saved inside `runs/`.

#### 🔍 Evaluation

__Step 1:__  Make sure config.py file has `evaluate` set to `True` (you can set `gui` to `True` to visually see the simulation)

__Step 2:__  The results will be saved in the corresponding `runs/` subfolder as 4 json files corresponding to:
 - Real-world crosswalk configuration with unsignalized control
 - Crosswalk configuration from the trained design agent with:
     -  Unsignalized control for mid-block crosswalks
     -  Traditional fixed-time traffic signal control for mid-block crosswalks
     -  Trained control agent for mid-block crosswalks

__Step 3:__  At the end of evaluations, a plot file named `design_control_results.pdf` will be saved.

*Unsignalized cases will be visually indicated by all mid-block signals turning green (for both vehicles and pedestrians).

---

### 📝 Code Structure
The main implementation is organized as follows:
```
├── main.py                # Training and evaluation pipeline orchestrator
├── config.py              # Configuration and hyperparameter management
├── utils.py               # General utility functions and helpers
├── sweep.py               # WandB hyperparameter sweep
├── images/                # Reference images and figures
├── plots/
│    ├── training_plots.py  # Training-time visualization
│    ├── result_plots.py    # Post-training analysis plots
├── ppo/
│    ├── models.py          # Neural network policy architectures
│    ├── ppo.py             # PPO algorithm implementation
│    ├── ppo_utils.py       # Memory, normalizers, graph dataset
├── simulation/
│    ├── design_env.py      # Higher-level design environment
│    ├── control_env.py     # Lower-level control environment
│    ├── worker.py          # Parallel train/eval workers
│    ├── sim_setup.py       # Traffic light phases and crosswalk definitions
│    ├── env_utils.py       # Graph visualization and coordinate utilities
```
#### Some important parameters that you can change in the config.py file:

-   `gui: True`  to run the simulation with GUI.
-   `gpu: True`  to load the policy on GPU. Set to `False` if your system does not have a  GPU (everything should work even without a GPU).
-   `"total_timesteps"`: Total training timesteps, default=20000000.
-   `"max_timesteps"`: Maximum per episode simulation steps, default=360.
-   `"num_processes"`: Number of parallel lower-level (control agent) processes, default=10. Increase/ decrease according to your CPU.

---
#### ⚠️  Debugging
- If something fails, check the  `netconvert_log.txt`, `sumo_logfile.txt`  and  `sumo_errorlog.txt`  files in the  `runs/`  folder.

---
### 📖 Citation
If you find this work useful in your own research:
```
@misc{poudel2025decor,
  title={DeCoR: Design and Control Co-Optimization for Urban Streets Using Reinforcement Learning},
  author={Poudel, Bibek and Zhu, Lei and Heaslip, Kevin and Swaminathan, Sai and Li, Weizi},
  year={2026},
  note={Preprint},
}
```
