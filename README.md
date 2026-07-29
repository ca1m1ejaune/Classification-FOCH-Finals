<div align="center">

# Classification-FOCH-Finals

**Classification multi-classe de pathologies vocales — branche duale AST + WavLM**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-FFD21F?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-StratifiedKFold-F7931E?style=flat-square&logo=scikitlearn)
![CUDA](https://img.shields.io/badge/CUDA-AMP%20Enabled-76B900?style=flat-square&logo=nvidia)

</div>

Ce dépôt regroupe mes meilleurs notebooks de classification sur la base de données vocale de Foch. Les trois notebooks implémentent la **même architecture de fusion bi-branche** (AST + WavLM-Large) et les mêmes conventions que le dépôt `Classification-FOCH-Test`, mais correspondent à trois étapes successives d'amélioration : réglage des hyperparamètres, correction de la tête de fusion, puis ajout d'une augmentation de données par conversion de voix.

## Contenu

| Notebook | Description | Balanced Accuracy (OOF) | Macro-F1 (OOF) | Macro AUC (OOF) |
|---|---|---|---|---|
| `5_0_foch_test_AST_HP.ipynb` | Configuration de référence (tête de fusion large, `HIDDEN_DIM` non réduit) | 0.634 | 0.632 | 0.834 |
| `5_0_foch_test_AST_HP_fixed.ipynb` | Tête de fusion régularisée (dropout renforcé, `HIDDEN_DIM` réduit) — corrige le surapprentissage observé sur la version précédente | 0.644 | 0.642 | 0.849 |
| `5_0_foch_test_AST_VC.ipynb` | Même architecture régularisée + augmentation par voice conversion (Seed-VC), ciblée sur la classe minoritaire `fuite glottique` | **0.691** | **0.666** | **0.879** |

Toutes les métriques ci-dessus sont calculées **out-of-fold** (chaque enregistrement prédit une seule fois, par le modèle du pli où il sert de validation) sur une validation croisée stratifiée à 5 plis. Le notebook `AST_VC` est actuellement le meilleur modèle du dépôt, notamment sur le rappel de la classe `fuite glottique` (la plus sous-représentée, n = 26).

## Architecture

Le classifieur est un réseau **multi-branche configurable** : une branche audio-domaine (spectrale) au choix entre CNN14 (PANNs, entraînable) ou AST (Audio Spectrogram Transformer, gelé), combinée à une branche WavLM optionnelle (Base ou Large, gelée par défaut). Un unique drapeau `BACKBONE` (Cellule 1 de chaque notebook) sélectionne le preset ; la tête de fusion (`Dropout → Linear → LayerNorm → Dropout → ReLU → Linear`) s'adapte automatiquement à la dimension d'embedding résultante.

| Preset | Branche audio-domaine (AudioSet) | Branche WavLM | Dimension de fusion |
|---|---|---|---|
| `cnn14` | CNN14 entraînable (2048-d) | WavLM-Base gelé (768-d) | 2816-d |
| `ast` | AST gelé (768-d) | WavLM-Base gelé (768-d) | 1536-d |
| `ast_large` *(configuration active des 3 notebooks)* | AST gelé (768-d) | WavLM-Large gelé (1024-d) | 1792-d |
| `cnn14_solo` / `ast_solo` | CNN14 / AST seul | — | 2048 / 768-d |
| `wavlm_base` / `wavlm_large` | — | WavLM Base / Large gelé | 768 / 1024-d |

Les trois notebooks de ce dépôt utilisent le preset `ast_large` : AST et WavLM-Large restent tous deux **gelés** (utilisés comme extracteurs d'embeddings), seule la tête de fusion est entraînée — entre 230k et 921k paramètres entraînables selon la configuration, contre plus de 400 millions de paramètres au total. Chaque pli est entraîné avec une `FocalLoss` pondérée par classe (pour compenser le déséquilibre `PR`/`LMB`/`fuite glottique`/`Healthy`), un planificateur cosinus avec échauffement, la précision mixte (`autocast` + `GradScaler`) et un early stopping sur la perte de validation.

## Données

Les jeux de données ne sont **pas inclus** dans ce dépôt (données patients). Chaque notebook attend, en Cellule 1, les chemins suivants à adapter à votre machine :

- `CSV_PATH` — métadonnées vocales au format `orl_df_vowel_master.csv` (colonnes attendues telles que `File_Name`, `Last_Name`, `label_str`, etc.)
- `FOCH_AUDIO_DIR` — dossier des segments audio Foch (voyelle `/a/` soutenue, `.wav`)
- `SVD_HEALTHY_DIR` — dossier des enregistrements sains de la base Saarbrücken (SVD), utilisés comme groupe contrôle
- `AUG_DIR` / `AUG_MANIFESTS` — signaux augmentés et manifestes CSV correspondants (schéma : `filepath`, `File_Name`, `Last_Name`, `label_str`, `is_aug`), injectés **uniquement dans le train de chaque pli** pour éviter toute fuite d'information
- `AST_DIR`, `WAVLM_DIRS`, `CNN14_CKPT_PATH` — copies locales des poids pré-entraînés (voir Dépendances ci-dessous)

Le notebook `AST_VC` attend en plus un manifeste `manifest_train_bal_vc.csv`, généré à partir de `augmentation_manifest_seedvc.csv` (produit par un notebook externe `voice_conversion/seed_vc.ipynb`, non inclus ici) : la Cellule 1b fait le pont entre le schéma brut Seed-VC et le schéma attendu par la boucle d'entraînement.

La classe `Cancer` est volontairement exclue de l'entraînement (support insuffisant, n ≈ 26).

## Dépendances

Les notebooks reposent sur PyTorch, torchaudio, HuggingFace Transformers et scikit-learn — pas uniquement sur la pile scientifique classique (numpy/pandas/scikit-learn) qui figurait dans une version précédente de ce fichier.

1. Cloner le dépôt :

   ```bash
   git clone https://github.com/ca1m1ejaune/Classification-FOCH-Finals.git
   cd Classification-FOCH-Finals
   ```

2. Créer un environnement Python 3.10+ :

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
   pip install --upgrade pip
   ```

3. Installer PyTorch et torchaudio en fonction de votre version de CUDA (voir le sélecteur officiel sur [pytorch.org](https://pytorch.org/get-started/locally/)), par exemple :

   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. Installer les dépendances restantes :

   ```bash
   pip install transformers huggingface_hub scikit-learn pandas numpy matplotlib seaborn jupyter notebook
   ```

5. Installer **FFmpeg** séparément (backend de décodage audio utilisé par torchaudio) et vérifier qu'il est accessible dans le `PATH` — ce n'est pas un paquet pip. Sous Windows, la Cellule 1 ajoute automatiquement un chemin local au `PATH` si présent ; à adapter ou supprimer selon votre installation.

### Poids pré-entraînés

- **AST** (`ast-finetuned-audioset-10-10-0.4593`) et **WavLM** (Base et/ou Large) sont chargés depuis un dossier **local** dans les notebooks actuels (`AST_DIR`, `WAVLM_DIRS`). Téléchargez-les au préalable depuis HuggingFace Hub (`MIT/ast-finetuned-audioset-10-10-0.4593`, `microsoft/wavlm-base`, `microsoft/wavlm-large`) ou remplacez ces chemins par les identifiants Hub directement.
- **CNN14** (preset `cnn14`, non utilisé par les 3 notebooks de ce dépôt mais présent dans le code) tente un chemin local (`CNN14_CKPT_PATH`) puis, à défaut, un téléchargement automatique via `huggingface_hub.hf_hub_download` depuis `speechbrain/cnn14-esc50`. L'implémentation de CNN14 est recopiée localement dans le notebook (aucune dépendance au paquet `speechbrain` n'est nécessaire).

### Matériel recommandé

Un GPU CUDA est fortement recommandé (AMP + `GradScaler` supposent `cuda` disponible ; l'entraînement CPU des 5 plis serait très lent). Repères VRAM mesurés pour le preset `ast_large` (fenêtre audio 4 s) :

- 8 Go de VRAM : `BATCH_SIZE = 8` avec `ACCUM_STEPS = 2` (batch effectif 16)
- 16 Go de VRAM (ex. RTX 5070 Ti) : `BATCH_SIZE` 16-24, `ACCUM_STEPS = 1`

## Utilisation

1. Ouvrir le notebook souhaité et adapter les chemins de la **Cellule 1** (données, poids pré-entraînés, dossiers de sortie).
2. Lancer Jupyter :

   ```bash
   jupyter lab
   # ou
   jupyter notebook
   ```

3. Exécuter les cellules **dans l'ordre** (Cellule 1 à 8/9) : chaque notebook entraîne 5 modèles (un par pli de validation croisée), ce qui suppose un temps d'exécution non négligeable même sur GPU.
4. Les poids de modèle et les artefacts d'évaluation (tableau de bord PNG, métriques JSON, matrice de confusion CSV, métriques par pli CSV) sont sauvegardés respectivement dans les dossiers configurés par `WEIGHTS_DIR` et `METRICS_DIR`.

## Limites connues et pistes d'amélioration

- `Last_Name` est un identifiant patient approximatif (homonymes possibles) ; un identifiant patient dédié permettrait une déduplication plus fiable.
- La classe `fuite glottique` reste la plus difficile (support le plus faible) malgré la `FocalLoss` et l'augmentation Seed-VC ciblée.
- Pistes envisagées : dégel progressif partiel d'AST/WavLM après stabilisation de la tête, SpecAugment sur l'entrée fbank d'AST, ensembling des modèles de pli à l'inférence, réduction supplémentaire de la dimension de fusion (projection/PCA) si le head reste sujet au surapprentissage.

## Structure et conventions

Chaque notebook suit les mêmes structures et conventions que celles du dépôt `Classification-FOCH-Test` (prétraitement, entraînement, évaluation, visualisation) : configuration centralisée en début de notebook, `Dataset` bi-entrée, validation croisée stratifiée à 5 plis avec augmentation injectée uniquement côté train, et évaluation agrégée out-of-fold.

## Contribution

Les contributions et retours sont bienvenus. Ouvrez une issue pour proposer des améliorations ou signaler un problème.

## Licence

Aucune licence n'est spécifiée pour l'instant. Si vous souhaitez en ajouter une, merci d'ajouter un fichier `LICENSE`.

---

Contact : https://github.com/ca1m1ejaune
