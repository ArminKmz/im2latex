# im2latex

PyTorch implementation of [this paper](https://arxiv.org/abs/1609.04938v2) with minor variations for my deep learning course final project.

# Training and Parameters
Add dataset pathes to `params.py`, you can also change the network and optimization parameters in this file. To start training the model or resume from last saved checkpoint, run 

```
python main.py --train
```

To evaluate the model on test or validation set run

```
python main.py --evaluate --evalset [test or validation] --checkpoint path/to/checkpoint
```

if you don't provide `--checkpoint` argument, it will use last saved checkpoint.
