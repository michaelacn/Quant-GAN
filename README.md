# Quant GAN : Deep Generation of Financial Time Series

Cette implémentation s’appuie sur l’article **[Wiese et al., “Quant GANs: Deep Generation of Financial Time Series,” 2019](https://arxiv.org/abs/1907.06673)** et inclut également du code adapté de :  
- [ICascha/QuantGANs-replication](https://github.com/ICascha/QuantGANs-replication)  
- [LocusLab/TCN](https://github.com/locuslab/TCN)  
- Greg Ver Steeg (2015)
- [JamesSullivan/temporalCN](https://github.com/JamesSullivan/temporalCN/tree/main)

---

## Structure du projet


- **model/** : Scripts pour implémentation du model (sous PyTorch).  
- **preprocess/** : Scripts pour préparer les séries temporelles.  
- **saved_models/** : Répertoire pour sauvegarder les modèles entraînés au format JSON.  
- **train.ipynb** : Notebook principal pour entraîner le Quant GAN
- **preprocess.ipynb** : Explication détaillée du prétraitement de données.  
- **torch_model.ipynb** : Explication de l'architecture du model.  
- **requirements.txt** : Dépendances Python requises.

---

## Installation

1. **Cloner ce dépôt** :  
   ```bash
   git clone https://github.com/michaelacn/Quant-GAN
   cd Quant-GAN
   pip install -r requirements.txt
   ```
