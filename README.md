# ImprovedDiffusion_based_GuidedDiffusion
Classifier guidance with improved diffusion model.



```bash
cd path/to/improved_diffusion
```

Then follow the instructions of [Editing ImprovedDiffusion_based_GuidedDiffusion/improved_diffusion/README.md at main · opoopop/ImprovedDiffusion_based_GuidedDiffusion](https://github.com/opoopop/ImprovedDiffusion_based_GuidedDiffusion/edit/main/improved_diffusion/README.md), including the preparation of the dataset, training and sampling, pre-trained models.

Then:

```bash
cd path/to/guided-diffusion
pip install -e .
```
**Doing all the train of the model under the file of improved diffusion and all the sampling with classifier guidance under guided diffusion file**.

Modules of the classifier guidance has been  added on improved diffusion and classifier_sample.py  has load the modules from improved diffusion. Follow the instructions from [ImprovedDiffusion_based_GuidedDiffusion/guided-diffusion at main · opoopop/ImprovedDiffusion_based_GuidedDiffusion](https://github.com/opoopop/ImprovedDiffusion_based_GuidedDiffusion/tree/main/guided-diffusion) to sample the image. 



[ImprovedDiffusion_based_GuidedDiffusion/FID_test.ipynb at main · opoopop/ImprovedDiffusion_based_GuidedDiffusion](https://github.com/opoopop/ImprovedDiffusion_based_GuidedDiffusion/blob/main/FID_test.ipynb) include the notebook to calculate FID and other scores from npz_file.



[ImprovedDiffusion_based_GuidedDiffusion/command_helper.md at main · opoopop/ImprovedDiffusion_based_GuidedDiffusion](https://github.com/opoopop/ImprovedDiffusion_based_GuidedDiffusion/blob/main/command_helper.md) has all the commands need on the experiment part.

