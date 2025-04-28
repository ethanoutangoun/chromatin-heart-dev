# chromatin-heart-dev

Repository for Chromatin Analysis during Heart Development

[Documentation](https://jet-shop-359.notion.site/Chromatin-Heart-Bioinformatics-Steps-19201216da2880f68376c4193d9109f7)



## Connecting to Expanse

Do all environment and configurations steps from within the login node instead of computer nodes to avoid using credits while idle/installing dependencies.

### Terminal A

Load SLURM module to act as Client to Request Interactive Node

```bash
module load slurm
```

Request Interactive Computer Node

```bash
srun --partition=shared --account=slo105 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=2:00:00 --pty /bin/bash
```

Optionally request a GPU node (must optimize code for CUDA operations to see improvements)

```bash
srun \
  --partition=gpu-shared \
  --account=slo105 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gres=gpu:1 \
  --mem=64G \
  --time=2:00:00 \
  --pty /bin/bash
```

Load Modules

```bash
module purge
module load slurm cpu/0.15.4 gcc/10.2.0
```

Activate miniconda and env

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate chromatin38
```

Run Jupyter Server

```bash
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

### Terminal B - Tunnel from Laptop

```bash
# (replace exp-?-?? with the hostname from Terminal A, e.g. exp-1-13)
ssh -L 9999:exp-6-38:8888 [username]@login.expanse.sdsc.edu 
```

### Quick Commands

```bash
# ── Terminal A (Expanse) ──
ssh login.expanse.sdsc.edu
module load slurm
srun -p shared -A slo105 -N1 -n1 --cpus-per-task=4 --mem=64G -t2:00:00 --pty bash -l
module purge; module load slurm cpu/0.15.4 gcc/10.2.0
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate chromatin38
cd ~/temp_project/chromatin-heart-dev
jupyter lab --no-browser --port=8888 --ip=0.0.0.0

# ── Terminal B (Local) ──
ssh -N -L 9999:<compute-hostname>:8888 your_user@login.expanse.sdsc.edu

# ── Browser ──
http://localhost:9999/lab?token=<YOUR_TOKEN>
```
