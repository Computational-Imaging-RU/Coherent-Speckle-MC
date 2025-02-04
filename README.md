# Monte-Carlo based efficient Image reconstruction in coherent imaging with speckle noise

## Preview
#### There are 4 python files in this repo. The test data is under the /data/ folder.

- train_DIP_deblur_ite_MC.py: training DIP based PGD algorithm.

- function_grad.py: i) Forward operator in coherent imaging, and ii) implementation of Monte-Carlo sampling and conjugate gradient methods to avoid matrix inverse in MLE based optimization.

- decoder.py: basic network structures of the Deep Image Prior/Deep Decoder we use for projection.

- utils.py: all the other helper functions.

## Running the code

#### Run the MLE-based PGD algorithm (efficient Monte-Carlo and conjugate gradient methods) for recovering images:

```
python train_DIP_deblur_ite_MC.py
```

#### Specify the hyperparameters and experiment setting:

#### E.g., recover images from measurements with number of looks L=1, aperture percentage=0.8, additive noise level=50, Monte-Carlo samples=10:

```
python train_DIP_deblur_ite_MC.py --dataset 'Set11' --mask_rate 0.8 --add_std 0.2 --num_look 1 --lr_NN 1e-3 --lr_GD 0.01 --outer_ite 100 --MC True --num_ite_MC 10
```

## Relevant works on speckle noise

[1] Chen, Xi, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Novel approach to coherent imaging in the presence of speckle noise." Unconventional Imaging, Sensing, and Adaptive Optics 2024. Vol. 13149. SPIE, 2024. [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13149/1314908/Novel-approach-to-coherent-imaging-in-the-presence-of-speckle/10.1117/12.3027824.full)

[2] Chen, Xi, Zhewen Hou, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Bagged Deep Image Prior for Recovering Images in the Presence of Speckle Noise." Forty-first International Conference on Machine Learning (ICML 2024). [paper](https://openreview.net/pdf?id=IoUOhnCmlX)

[3] Chen, Xi, Zhewen Hou, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Multilook compressive sensing in the presence of speckle noise." In NeurIPS 2023 Workshop on Deep Learning and Inverse Problems. 2023. [paper](https://openreview.net/forum?id=G8wMnihF6E)

[4] Zhou, Wenda, Shirin Jalali, and Arian Maleki. "Compressed sensing in the presence of speckle noise." IEEE Transactions on Information Theory 68.10 (2022): 6964-6980. [paper](https://ieeexplore.ieee.org/abstract/document/9783054)
