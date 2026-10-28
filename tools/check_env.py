import subprocess

import torch
import transformers

n = []

n.append(f'torch: {torch.__version__}\n')
n.append(f'transformers: {transformers.__version__}\n' )
n.append(f'CUDA Runtime: {torch.version.cuda}\n')
n.append(f'GPU Available: {torch.cuda.is_available()}\n')
n.append(f'\n{subprocess.run("nvidia-smi", capture_output=True, text=True).stdout}')

with open('notes/environment.txt', 'w') as f:
    f.writelines(n)
    f.close()
