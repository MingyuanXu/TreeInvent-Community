# TreeInvent-Community
Tree-Invent examples for the various drug design strategy

![cover image](./pics/TOC.png)

## Description
TreeInvent-Community shares the examples for various drug design strategy including the scaffold hopping, decoration and linker generation with or without topology constrains. By integrating the RL-algorithm, Tree-Invent could active the ultra-fast drug design with the desired structure or property.
## examples
Basic_train_examples: it is basic example of training the prior Tree-Invent model, including the dataset creation, basic hyper-parameter setting and unconstrained sampling.\n 
Finetune_examples: it is a transfer-learing examples for finetuning the prior Tree-Invent model, Tree-Invent will quickly overfits the finetune datasets in few epochs, it would be necessary to increase the temp_factor for a better diversity of samplings.\n
Finetune_RL_examples: Since the transfer-learning will quickly lead the Tree-Invent to overfitting and increasing the temp-factor will decrease the quality of samplings. Tree-Invent allows perform reinforcement learning with the finetuned prior model for much better diversity and quality of samplings. \n
Activity_RL_examples: it is an example of basic RL with QSAR SVC activity models for DRD2 target. \n
Linker_generation_examples: it is an example for tree-constrained linker design with RL for S1PR1 \n
Scaffold_decoration_examples: it is an example for tree-constrained scaffold_decoration with RL for ADAM17. \n 
Rocs_shape_examples: it is an basic example for 3D shape-based RL-learning with ROCS suites.\n
Vina_Dock_example: it is an example for Vina docking score driven RL-learning for a given target.\n 
Glide_Dock_example: it is an example for Glide docking score driven RL-learning for a given target.\n 
Glide_rocs_example: it is an example for both Glide docking score and rocs shape driven RL-learning for a given target. \n

## Contributors:
[@Mingyuan Xu](https://github.com/MingyuanXu)

## Examples
* Tree-Invent suite are avaliable from https://github.com/MingyuanXu/Tree-Invent.
