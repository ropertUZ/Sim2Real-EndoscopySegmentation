# Sim2Real in endoscopy segmentation with a novel structure aware image translation
**Authors**: Clara Tomasini, Luis Riazuelo, Ana C. Murillo

### Related Publications:
Tomasini, Clara, Luis Riazuelo, and Ana C. Murillo. "Sim2Real in Endoscopy Segmentation with a Novel Structure Aware Image Translation." International Workshop on Simulation and Synthesis in Medical Imaging. Cham: Springer Nature Switzerland, 2024. [**PDF**](https://arxiv.org/pdf/2505.02654)
```
@inproceedings{tomasini2024sim2real,
  title={Sim2Real in Endoscopy Segmentation with a Novel Structure Aware Image Translation},
  author={Tomasini, Clara and Riazuelo, Luis and Murillo, Ana C},
  booktitle={International Workshop on Simulation and Synthesis in Medical Imaging},
  pages={89--101},
  year={2024},
  organization={Springer}
}
```

This software has been trained and evaluated with a few sequences from the EndoMapper dataset, as described in:

Azagra, Pablo, et al. "Endomapper dataset of complete calibrated endoscopy procedures." Scientific Data 10.1 (2023): 671.

# 1. License
This repository is released under AGPLv3 license.
### Third-party code
This repository is built on a fork of project [**CycleGAN-and-pix2pix**](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git) (with BSD License), the official implementation of the paper:
Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.

# 2. Prerequisites
The software has been tested on **Ubuntu 20.04** and uses [Python](https://www.python.org). **Required 3.X**.

# 3. Proposed image translation pipeline
![fig1](https://github.com/user-attachments/assets/9759c20e-5fa4-44e7-acd9-ad12428cd70a)

# 4. How to run
Weights for our modified CycleGAN network trained on simulated images from VR-CAPS and real images from EndoMapper can be found here: [**Weights**](https://drive.google.com/drive/folders/1oS8HHqoYd5FFLuCjmFix1r9CrTLkpWMy?usp=drive_link)
# 5. Fold segmentation annotations and results
Data used in our paper for train and test can be found at the following links, containing RGB images (folder *original.zip*) with binary (folder *gt.zip*) and instance (folder *inst.zip*) ground-truth segmentation masks.

- [**Simulated test set adapted from VR-CAPS**](https://drive.google.com/drive/folders/1S-hbntHkmbOEWNXvtIMvhwts2mE3K2TL?usp=drive_link): simulated (folder *original.zip*) and simulated with added realistic texture (folder *original_aug.zip*) images, ground-truth binary (folder *gt.zip*) and instance (folder *inst.zip*) segmentations
- [**Simulated train set adapted from VR-CAPS**](https://drive.google.com/drive/folders/13Q8GTTmJw-6nyETqcnyMUhWqcUcOXqLg?usp=drive_link): RGB images (folder *original.zip*), depth maps (*depth.zip*), binary (*gt.zip*) and instance (*inst.zip*) segmentation masks
- [**Real test set from EndoMapper dataset**](https://drive.google.com/drive/folders/1L1tgOwAGnCCba30X9yPw_O90LapxlvD5?usp=share_link)
- [**Real train set from EndoMapper dataset**](https://drive.google.com/drive/folders/1EJOCb66-Rjt8Xi5c9HbZhTN50DR9L1W_?usp=drive_link): RGB images (*original.zip*)
  
![res_vis](https://github.com/user-attachments/assets/cb8e1f21-6f86-4493-bcbb-fde5873f7a9f)
