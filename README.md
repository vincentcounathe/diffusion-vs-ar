# Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning


This repository contains code for training and evaluating the models in the paper [Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning](https://arxiv.org/abs/2410.14157).

<p align = "center">
<img src="simple_task.png" width="95%" alt="simple_task" align=center />
</p>

- Autoregressive language models, despite their impressive capabilities, struggle with complex reasoning and long-term planning tasks. Can we go beyond autoregression for these challanges?
- First, what is planning essentially? We design a straightforward task to minimally illustrate planning, where we can also control the extent of planning through a term called Planning Distance. We find AR struggles a lot on this simple task. 
- Then, we delve into the comparison between the objective of autoregression and discrete diffusion, and demonstrate how discrete diffusion models effectively learn difficult subgoals that elude autoregressive models. 
- Based on above, we further introduce Multi-granularity Diffusion Modeling (MDM), which prioritizes subgoals based on difficulty during learning. We find MDM significantly outperforms AR on various more complex reasoning and planning challanges.


## Setup
All required packages can be found in requirements.txt. You can install them in a new environment with
```
conda create -n diffusion python=3.9
conda activate diffusion
git clone git@github.com:HKUNLP/diffusion-vs-ar.git

cd diffusion-vs-ar
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
Training and evaluation commands are provided under the `scripts` directory. Download data from [here](https://drive.google.com/file/d/1b0OIlYL76rVVuNYIfIb-L_Ptdg6k_y0c/view?usp=sharing) first.
The synthetic planning dataset can also be generated using `data/synthetic_graph.py` script.

To stream metrics to Weights & Biases instead of the old disabled default, make sure the relevant environment variables are set before launching:

```
export WANDB_API_KEY=xxxxxxx
export WANDB_PROJECT=diffusion-vs-ar
export WANDB_DIR=/share/.../wandb   # optional cache location
```

```
# run AR (training from scratch)
bash scripts/sudoku/train-sft.sh

# run AR (finetuning from LLaMA)
bash scripts/sudoku/train-sft-llama-7b.sh

# run Diffusion (training from scratch)
bash scripts/sudoku/train-mdm.sh
```
(📌check our work on scaling diffusion langauge model by adapting from LLaMA at https://github.com/HKUNLP/DiffuLLaMA)

For experiment on different model size, change `--model_name_or_path` to `model_config_tiny` (~6M), `model_config` (~85M) or `model_config_medium` (~303M). A slightly larger learning rate (i.e., 1e-3) is used for the tiny model.

For experiment on different datasets, change `--dataset` (dataset name in `data/dataset_info.json`) and adjust `--cutoff_len` (make sure equal or larger than the largest token length on that dataset). For AR, make sure the `--max_new_tokens` during generation is also larger than that seen in the training time. Here are the `cutoff_len` and `--max_new_tokens` used in the paper:

|                      | Minimal Planning | Countdown 3 | Countdown 4 | Countdown 5 | Sudoku | 3-SAT 5v | 3-SAT 7v | 3-SAT 9v |
|----------------------|:----------------:|:-----------:|:-----------:|:-----------:|:------:|:--------:|:--------:|:--------:|
| cutoff_len           |               75 |      37     |      64     |          74 |    164 |      258 |      285 |      325 |
| max_new_tokens (sft) | 24               |          24 |          32 |          54 |     82 |       10 |       14 |       18 |

Please refer to Appendix C.2 for other illustraion of implementation details.

## Information-Gain Critic & Ordered Decoding

1. **Stage 1:** Train the diffusion model as usual (see commands above) and keep the checkpoint.
2. **Stage 2:** Fit the critic with the frozen denoiser:

```bash
python training/train_critic.py \
  --stage mdm \
  --model_name_or_path model_config_tiny \
  --dataset sudoku_train \
  --cutoff_len 164 \
  --do_train \
  --output_dir output/sudoku/critic \
  --learning_rate 1e-4 \
  --num_train_epochs 20 \
  --per_device_train_batch_size 16
```

The script saves `info_gain_critic.pt`, which contains the critic weights and feature configuration.
Use the same stage-1 test split for evaluation (e.g., rerun with `--dataset sudoku_test` or via `scripts/sudoku/eval_mdm.sbatch`) and drop the legacy `critic_{train,test}` CSVs.

3. **Inference:** Enable greedy information-gain decoding with the trained critic by adding

```
--use_info_gain_ordering \
--critic_checkpoint output/sudoku/critic/info_gain_critic.pt \
--info_gain_alpha 0.5 \
--info_gain_tau_util 0.0 \
--info_gain_tau_conf 0.0
```

to the usual diffusion sampling command. The sampler loads the critic and greedily reveals tokens based on predicted utility while respecting the readiness thresholds.

## Citation
If you find our code or data helpful, please cite us as follows 
```
@article{ye2024beyond,
  title={Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning},
  author={Ye, Jiacheng and Gao, Jiahui and Gong, Shansan and Zheng, Lin and Jiang, Xin and Li, Zhenguo and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2410.14157},
  year={2024}
}
```

The code framework is adapted from [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory), thanks for their great work.
