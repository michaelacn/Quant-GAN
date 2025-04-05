# Quant GAN : Deep Generation of Financial Time Series

Cette implémentation s’appuie sur l’article **[Wiese et al., “Quant GANs: Deep Generation of Financial Time Series,” 2019](https://arxiv.org/abs/1907.06673)** et inclut également du code adapté de :  
- [ICascha/QuantGANs-replication](https://github.com/ICascha/QuantGANs-replication)  
- [LocusLab/TCN](https://github.com/locuslab/TCN)  
- Greg Ver Steeg (2015)
- [JamesSullivan/temporalCN](https://github.com/JamesSullivan/temporalCN/tree/main)

---

## Structure du projet


- **model/** : Scripts .py pour implémentation du model (sous PyTorch).  
- **preprocess/** : Scripts .py pour préparer les séries temporelles.  
- **saved_models/** : Répertoire pour sauvegarder les modèles entraînés au format JSON.  
- **train.ipynb** : Notebook principal pour entraîner le Quant GAN
- **preprocess.ipynb** : Explication détaillée du prétraitement de données.  
- **torch_model.ipynb** : Explication détaillée de l'architecture du model.  
- **requirements.txt** : Dépendances Python requises.

---

## Installation

1. **Installer ou mettre à jour Conda** (si vous ne l’avez pas déjà fait). Vous pouvez consulter la documentation de [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou [Anaconda](https://www.anaconda.com/products/distribution) pour obtenir des instructions.

2. **Cloner ce dépôt et installer les dépendances** :
   ```bash
   git clone https://github.com/michaelacn/Quant-GAN.git
   cd Quant-GAN
   conda create --name quantgan python=3.11 -c https://conda.anaconda.org/conda-forge/ -y
   conda activate quantgan
   pip install -r requirements.txt
   python -m ipykernel install --user --name quantgan --display-name "Python (quantgan)"
   ```

Vous disposez maintenant d’un environnement conda nommé `quantgan` avec tous les paquets requis installés. N’hésitez pas à rafraîchir votre liste de noyaux (kernels) pour qu’il apparaisse.
