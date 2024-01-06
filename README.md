# MFPO
This work "Federated Offline Policy Optimization with
Dual Regularization" has been submitted in INFOCOM 2024.
## :page_facing_up: Description
doubly regularized federated offline policy optimization (DRPO), that leverages dual regularization, one based on the local behavioral state-action distribution and the other on the global aggregated policy.
Specifically, the first regularization can incorporate conservatism into the local learning policy to ameliorate the effect of extrapolation errors. The second can confine the local policy around the global policy to impede over-conservatism induced by the first regularizer and enhance the utilization of the aggregated information.
## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.8.1](https://pytorch.org/)
- [MuJoCo == 2.3.6](http://www.mujoco.org) 
- NVIDIA GPU (RTX A6000) + [CUDA 11.1](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/HansenHua/DRPO-INFOCOM24.git
    cd MFPO-Online-Federated-Reinforcement-Learning
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :zap: Quick Inference

Get the usage information of the project
```bash
python main.py -h
```

Test the trained models provided in [DRPO-doubly regularized federated offline policy optimization](https://github.com/HansenHua/DRPO-INFOCOM24/tree/main).
```
python main.py halfcheetah-medium-expert-v2 DRPO test
```
## :computer: Training

We provide complete training codes for MFPO.<br>
You could adapt it to your own needs.

	```
    python main.py halfcheetah-medium-expert-v2 DRPO train
	```

## :checkered_flag: Testing
### Testing
	```
	python main.py CartPole-v1 MFPO test
	```

### Open issues:
Models for testing is not updated now 

## :e-mail: Contact

If you have any question, please email `xingyuanhua@bit.edu.cn`.
