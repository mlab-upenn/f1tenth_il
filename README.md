# F1TENTH Imitation Learning

This repository is forked from [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) for developing imitation learning methods on F1TENTH platform.

You can find the documentation of the F1TENTH gym environment [here](https://f1tenth-gym.readthedocs.io/en/latest/).

## Quickstart
- Clone this repository
```bash
git clone git@github.com:mlab-upenn/f1tenth_il.git
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
python inference.py --training_config=il_config.yaml --model_path=logs/HGDAgger/HGDAgger_svidx_0_dist_312_expsamp_5846.pkl
```

### Bootstrapping PPO
- Train an initial model following the steps in the [Training](#training) section
- Copy the model from ```imitation_learning/logs/<algorithm name>``` to ```reinforcement_learning/initial_models```
- Edit line 21 of ```reinforcement_learning/train_f1tenth_ppo.py``` to the proper directory of the initial model

