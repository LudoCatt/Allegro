#!/usr/bin/env bash
#SBATCH --job-name=finetune
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --gres=gpumem:80g
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=20G
#SBATCH --output=logs/%x_%j_out.txt
#SBATCH --error=logs/%x_%j_err.txt

# ── 1 · toolchain ───────────────────────────────────────────────────────────────
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.4.1

# ── 2 · conda env ──────────────────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ── 3 · environment ────────────────────────────────────────────────────────────
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TRITON_CACHE_DIR=/cluster/scratch/$USER/triton_cache
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_API_KEY=...
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 4 · training launch ────────────────────────────────────────────────────────
conda run -n allegro --no-capture-output \
  python -u -m accelerate.commands.launch \
    --num_machines 1 \
    --num_processes 2 \
    --machine_rank 0 \
    --config_file config/accelerate_config.yaml \
    train_ti2v.py \
      --project_name AllegroTI2V_Finetune_88x720p \
      --dit_config /cluster/scratch/lcattaneo/Allegro/transformer/config.json \
      --dit /cluster/scratch/lcattaneo/Allegro/transformer/ \
      --tokenizer /cluster/scratch/lcattaneo/Allegro/tokenizer \
      --text_encoder /cluster/scratch/lcattaneo/Allegro/text_encoder \
      --vae /cluster/scratch/lcattaneo/Allegro/vae \
      --vae_load_mode encoder_only \
      --dataset ti2v \
      --data_dir /cluster/scratch/lcattaneo/Videos/ \
      --meta_file data.parquet \
      --sample_rate 2 \
      --num_frames 64 \
      --max_height 720 \
      --max_width 1280 \
      --hw_thr 1.0 \
      --hw_aspect_thr 1.5 \
      --dataloader_num_workers 4 \
      --gradient_checkpointing \
      --train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --max_train_steps 10000 \
      --learning_rate 1e-5 \
      --lr_scheduler constant \
      --lr_warmup_steps 0 \
      --mixed_precision bf16 \
      --report_to wandb \
      --allow_tf32 \
      --model_max_length 512 \
      --cfg 0.1 \
      --checkpointing_steps 100 \
      --seed 42 \
      --i2v_ratio 1.0 \
      --interp_ratio 0.0 \
      --v2v_ratio 0.0 \
      --default_text_ratio 0.5 \
      --resume_from_checkpoint latest \
      --output_dir /cluster/scratch/lcattaneo/output/AllegroTI2V_Finetune_88x720p
