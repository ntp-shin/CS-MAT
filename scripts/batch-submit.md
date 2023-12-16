- gen
```bash
sbatch -w gpu01 --gpus=1 --cpus-per-gpu=16 --mem-per-gpu=10GB -o std/gen-out -e std/gen-err scripts/gen-request.script
```

- train
```bash
sbatch -w gpu01 --gpus=2 --cpus-per-gpu=32 --mem-per-gpu=20GB -o std/train-out -e std/train-err scripts/train-request.script
```