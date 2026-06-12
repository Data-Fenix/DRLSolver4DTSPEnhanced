# Solving Dynamic Traveling Salesman Problems With Deep Reinforcement Learning
This code solves dynamic trvaeling saleman problem with deep reinforcement learning. For more details, please see our paper [Solving Dynamic Traveling Salesman Problems With Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9537638) which has been accepted at IEEE-TNNLS. If this code is useful for your work, please cite our paper:

```
@article{zhang2021solving,
  title={Solving Dynamic Traveling Salesman Problems With Deep Reinforcement Learning},
  author={Zhang, Zizhen and Liu, Hong and Zhou, MengChu and Wang, Jiahai},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
```

## Dependencies

* python = 3.6.3
* NumPy
* Scipy
* PyTorch = 1.7
* tensorboard_logger

## If you have a higher Python version:

py -3.8 -m venv .venv
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install "numpy==1.21.6" "scipy==1.7.3" tensorboard_logger


## Quick Start

For training DTSP instances with 19 customers and using rollout as REINFORCE baseline with model M1:

```
python m1/train.py --baseline rollout --graph_size 19

python m1/train.py --baseline rollout --graph_size 19 --n_epochs 100 --run_name baseline_model # 100 epochs and it mentioned in the options.py file (line 37)
```
## Experiments
For training DTSP instances with 19 customers and using rollout as REINFORCE baseline with MLP+ CostAware Gating:

```
python m1/train.py --baseline rollout --graph_size 19 --n_epochs 10 --run_name gating_nearest_neighbor_mlp --use_cost_aware_gating --heuristic_type nearest_neighbor --lambda_heuristic 1.0 --use_step_mlp --step_mlp_dim 64 
```

For testing DTSP instances with 19 customers with a trained model M1:

```
python m1/test.py --baseline rollout --graph_size 19 --resume trained_models/m1/normal_19.pt
# use this below
python m1/test.py --load_path outputs/tsp_19/baseline_model_20251005T124249/epoch-9.pt --graph_size 19 --baseline rollout


## comparing models
python m1/test.py --graph_size 19 --baseline rollout --compare_models outputs/tsp_19/baseline_20251030T082105/epoch-99.pt outputs/tsp_19/mlp_gating_linear_model_20251022T113722/epoch-9.pt --compare_names Baseline "MLP+Gating"
```
Have some issues with heusristic model and need to fix it in the test.py script because cost is really low

For testing DTSP instances with 19 customers with a trained model M2:

```
python m2/test.py --baseline rollout --graph_size 19 --resume trained_models/m2/normal_19.pt
```

## Hyper parameter Tunning

Once you identify which heuristic(s) perform best, do a lambda sweep: Try different lambda values for the best heuristic
```
for lambda_val in 0.1 0.5 1.0 2.0 5.0; do
    python m1/train.py --baseline rollout --graph_size 19 --n_epochs 10 \
        --run_name gating_linear_time_lambda_${lambda_val} \
        --use_cost_aware_gating --heuristic_type linear_time \
        --lambda_heuristic ${lambda_val}
done
```
## Acknowledgements

Our code is adpated from [https://github.com/wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route).
