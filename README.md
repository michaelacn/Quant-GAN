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

1. **Install or update Conda** (if you haven’t already). You can refer to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) for instructions.

2. **Clone this repository and install dependencies**:
   ```bash
   git clone https://github.com/michaelacn/Quant-GAN
   cd Quant-GAN
   conda create --name quantgan python=3.11 -c https://conda.anaconda.org/conda-forge/ -y
   conda activate quantgan
   pip install -r requirements.txt
   ```

You should now have a conda environment named `quantgan` with all required packages installed.
