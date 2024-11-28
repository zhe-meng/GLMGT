
# GLMGT

# Global–Local Multigranularity Transformer for Hyperspectral Image Classification (JSTARS2025)

PyTorch implementation of Global–Local Multigranularity Transformer for Hyperspectral Image Classification.

# Basic Usage

```
import torch
from GLMGT import GLMGT
# Take the Indian Pines dataset as an example, the number of classes and spectral channels are 16 and 200, respectively.
model = GLMGT(in_chans=200, num_classes=16, patch_size=11)
model.eval()
print(model)
input = torch.randn(100, 200, 11, 11)
y = model(input)
print(y.size())
```

# Paper

[Global–Local Multigranularity Transformer for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/10746388)

If you find this code to be useful for your research, please consider citing.

```
@article{meng2024global,
  title={Global-Local MultiGranularity Transformer for Hyperspectral Image Classification},
  author={Meng, Zhe and Yan, Qian and Zhao, Feng and Chen, Gaige and Hua, Wenqiang and Liang, Miaomiao},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025},
  volume={18},
  pages={112-131},
  publisher={IEEE}
}
```


