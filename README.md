# Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks

An official implementation of "[Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks](https://arxiv.org/abs/2405.12295)" (ECAI 2024).

## Installation

```bash
pip install -r requirements.txt
```

## Experiments

```
python main.py --task=embedding --surrogate=gat --target=gin --dataset=citeseer_full --type=i --ratio_q=1.0
```

Note that we use parameters in our paper:
```
--task: ['embedding', 'prediction', 'projection']
--dataset: ['dblp', 'pubmed', 'citeseer_full', 'coauthor_phy', 'acm', 'amazon_photo']
--target: ['gat', 'gin', 'sage']
--surrogate: ['gat', 'gin', 'sage']
--type:   ['i', 'ii']
--ratio_q: A float between 0 and 1.
```

##Ackno
This code is based on implementations: https://github.com/xinleihe/GNNStealing and https://github.com/liun-online/SpCo.


## Cite this project

```bibtex
@inproceedings{podhajski2024efficient,
    title = {Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks},
    author = {Marcin Podhajski and 
              Jan Dubiński and 
              Franziska Boenisch and
              Adam Dziedzic and
              Agnieszka Pręgowska and
              Tomasz P. Michalak},
    booktitle = {The 27th European Conference on Artificial Intelligence (ECAI 2024)},
    series = {Frontiers in Artificial Intelligence and Applications},
    publisher = {IOS Press},
    year = {2024}
}
```