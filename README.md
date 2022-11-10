![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
# F1TENTH Imitation Learning

This repository is forked from [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) for developing imitation learning methods on F1TENTH platform.

You can find the documentation of the F1TENTH gym environment [here](https://f1tenth-gym.readthedocs.io/en/latest/).

## Quickstart
Clone this repository
```bash
git clone git@github.com:mlab-upenn/f1tenth_il.git
```

Navigate to the root directory of this project
```bash
cd f1tenth_il
```

Create a new conda environment with Python 3.8
```bash
conda create -n f110_il python=3.8
```

Activate the environment
```bash
conda activate f110_il
```

Install pip
```bash
conda install pip  
```

Install the dependencies for F1TENTH gym.
```bash
pip install -e .
```

Install other dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Training
Navigate to the imitation learning folder
```bash
cd "imitation learning"
```

Execute the training script
```bash
python train.py --algorithm=<algorithm name> --training_config=<yaml file location>
```


### Inference
Navigate to the imitation learning folder
```bash
cd "imitation learning"
```

Execute the inference script
```bash
python inference.py --training_config=<yaml file location> --model_path=<model path>
```
