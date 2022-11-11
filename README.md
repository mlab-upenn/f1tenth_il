# F1TENTH Imitation Learning

This repository contains code associated with [A Benchmark Comparison of Imitation Learning-based Control Policies for Autonomous Racing
](https://arxiv.org/abs/2209.15073)

## Quickstart
- Clone this repository
```bash
git clone -b bootstrap_rl https://github.com/mlab-upenn/f1tenth_il.git
```

- Navigate to the root directory of this project
```bash
cd f1tenth_il
```

- Create a new conda environment with Python 3.8
```bash
conda create -n f110_il python=3.8
```

- Activate the environment
```bash
conda activate f110_il
```

- Install pip
```bash
conda install pip  
```

- Install the dependencies for F1TENTH gym.
```bash
pip install -e .
```

- Install other dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Training
- Navigate to the imitation learning folder
```bash
cd imitation_learning
```

- Execute the training script
```bash
python train.py --algorithm=<algorithm name> --training_config=<yaml file location>
```

- Example:
```bash
python train.py --algorithm=hg-dagger --training_config=il_config.yaml
```

- The output models can be found under ```imitation_learning/logs/<algorithm name>```

### Inference
- Navigate to the imitation learning folder
```bash
cd imitation_learning
```

- Execute the inference script
```bash
python inference.py --training_config=<yaml file location> --model_path=<model path>
```

- Example:
```bash
python inference.py --training_config=il_config.yaml --model_path=logs/HGDAgger_svidx_1_dist_300_expsamp_36761.pkl
```

### Bootstrapping PPO
- Train an initial model following the steps in [Training](#training) section
- Copy the model from ```imitation_learning/logs/<algorithm name>``` to ```reinforcement_learning/initial_models```
    - A pre-trained HG-DAgger model is already in this directory
- Edit line 21 of ```reinforcement_learning/train_f1tenth_ppo.py``` to the proper directory of the initial model
- Execute the PPO training script
```bash
python train_f1tenth_ppo.py
```
    

