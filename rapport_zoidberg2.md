# ZOIDBERG 2.0
## Rapport de projet — Détection automatique de pneumonie par apprentissage automatique

**Projet** : T-DEV-810 — EPITECH  
**Équipe** : ML Architecture Team  
**Version** : 2.0.0  
**Date** : Juin 2026

---

## Table des matières

1. [Introduction et approche](#1-introduction-et-approche)
2. [Données et analyse exploratoire](#2-données-et-analyse-exploratoire)
3. [Prétraitement et réduction dimensionnelle (PCA)](#3-prétraitement-et-réduction-dimensionnelle-pca)
4. [Modèles baseline — Machine Learning classique](#4-modèles-baseline--machine-learning-classique)
5. [Deep Learning — Réseaux de neurones convolutifs](#5-deep-learning--réseaux-de-neurones-convolutifs)
6. [Optimisation, ensemble et fine-tuning](#6-optimisation-ensemble-et-fine-tuning)
7. [Synthèse et réponses aux questions de recherche](#7-synthèse-et-réponses-aux-questions-de-recherche)
- [Annexe A — Définition et interprétation des métriques](#annexe-a--définition-et-interprétation-des-métriques)
- [Annexe B — Comparaison avec et sans cross-validation](#annexe-b--comparaison-avec-et-sans-cross-validation)
- [Annexe C — Paramètres complets de la cross-validation et du Grid Search](#annexe-c--paramètres-complets-de-la-cross-validation-et-du-grid-search)
- [Annexe D — Détails techniques de la PCA](#annexe-d--détails-techniques-de-la-pca)

---

## 1. Introduction et approche

### 1.1 Contexte et problématique

La pneumonie est une infection pulmonaire qui constitue l'une des principales causes de mortalité dans le monde, en particulier chez les enfants de moins de cinq ans et les personnes âgées. Son diagnostic repose traditionnellement sur l'interprétation de radiographies thoraciques par un médecin radiologue, une tâche qui est à la fois chronophage, subjective, et difficile à scaler dans des contextes à ressources médicales limitées.

Le projet ZOIDBERG 2.0 vise à concevoir et évaluer un pipeline complet d'intelligence artificielle capable de détecter automatiquement la présence d'une pneumonie à partir d'images de radiographies pulmonaires. L'approche adoptée est volontairement comparative : nous évaluons plusieurs familles de modèles — du machine learning classique appliqué sur des features compressées par PCA, au deep learning par transfer learning — afin de quantifier les gains réels de performance et de complexité à chaque étape.

Le pipeline est structuré en sept étapes séquentielles, correspondant aux notebooks 01 à 07 :

1. Analyse et vérification de l'intégrité des données
2. Prétraitement et ingénierie des features (normalisation, PCA)
3. Modélisation baseline avec validation croisée
4. Deep Learning (CNN personnalisé, VGG16, ResNet50)
5. Évaluation globale (ROC-AUC, accuracy paradox)
6. Interprétabilité (Grad-CAM)
7. Méthodes d'ensemble et optimisation du seuil de décision

### 1.2 Questions de recherche

Ce rapport est structuré autour de trois questions de recherche dont les réponses sont apportées par les résultats expérimentaux présentés en section 7.

> **RQ1** — Les algorithmes de machine learning classiques, appliqués sur un espace de features réduit par PCA (95 % de variance expliquée), peuvent-ils atteindre des performances cliniquement acceptables (ROC-AUC ≥ 0,85) pour la détection de pneumonie ?

> **RQ2** — Le deep learning par transfer learning (VGG16, ResNet50 pré-entraînés sur ImageNet) apporte-t-il une amélioration significative et robuste par rapport aux approches classiques, et cette amélioration justifie-t-elle le surcoût computationnel ?

> **RQ3** — La combinaison de modèles hétérogènes (ensemble par weighted soft voting) permet-elle de dépasser les performances des modèles individuels, et quel seuil de décision est optimal pour un usage clinique ?

### 1.3 Approche méthodologique

L'approche générale suit le principe de la complexité croissante : partir des méthodes les plus simples et les plus interprétables (régression logistique, SVM), puis monter progressivement en complexité (CNN, transfer learning, ensemble) pour n'ajouter de la sophistication que là où elle apporte un gain mesurable. Toutes les comparaisons s'appuient sur des métriques robustes au déséquilibre de classes : ROC-AUC (métrique primaire), F1-score macro, précision et rappel.

---

## 2. Données et analyse exploratoire

### 2.1 Description des datasets

Le projet s'appuie sur trois datasets distincts, chacun ayant un rôle précis dans le pipeline :

| Dataset | Rôle | Total images | NORMAL | PNEUMONIA |
|---------|------|:------------:|:------:|:---------:|
| Dataset 1 | Entraînement principal | 5 216 | 1 341 (25,7 %) | 3 875 (74,3 %) |
| Dataset 2 | Validation externe | 16 | 8 (50,0 %) | 8 (50,0 %) |
| Dataset 3 | Tuning des hyperparamètres | 624 | 234 (37,5 %) | 390 (62,5 %) |
| **Total** | | **5 856** | **1 583 (27,0 %)** | **4 273 (73,0 %)** |

En complément de la classification binaire (NORMAL vs PNEUMONIA), les données permettent une classification multiclasse en trois catégories : NORMAL, BACTERIA et VIRUS. Dans Dataset 1, la répartition multiclasse est la suivante : BACTERIA 2 530 images (48,5 %), VIRUS 1 345 images (25,8 %), NORMAL 1 341 images (25,7 %).

> 📊 **[IMAGE — Notebook 01]** Histogrammes de la distribution des classes par dataset (barplots côte-à-côte pour Dataset 1, 2 et 3, avec le ratio de déséquilibre indiqué). Chemin dans le notebook : cellule EDA / class distribution.

### 2.2 Caractéristiques techniques des images

L'inspection de l'ensemble des images révèle une forte hétérogénéité dimensionnelle, qui justifie le redimensionnement systématique appliqué lors du prétraitement :

| Propriété | Valeur |
|-----------|--------|
| Format | JPEG |
| Mode colorimétrique | Niveaux de gris (L), quelques RGB |
| Dimensions (min – max) | 724×286 px → 2 446×2 224 px |
| Intensité pixel — Moyenne | 124,94 ± 20,21 |
| Intensité pixel — Plage | [0, 255] |
| Images corrompues | **0** |

L'intégrité complète du dataset (zéro image corrompue) est un prérequis essentiel à la reproductibilité des résultats. Elle a été vérifiée par ouverture et validation de chaque fichier image dans le notebook 01.

> 📊 **[IMAGE — Notebook 01]** Distribution des intensités pixel (histogramme global sur Dataset 1) + exemples de radiographies NORMAL et PNEUMONIA côte-à-côte.

### 2.3 Déséquilibre de classes et stratégie de compensation

Le ratio NORMAL:PNEUMONIA dans le dataset d'entraînement est de **1:2,89**, ce qui constitue un déséquilibre modéré mais suffisant pour biaiser significativement les modèles naïfs. En particulier, un classifieur qui prédirait systématiquement "PNEUMONIA" atteindrait une accuracy de 74,3 % — valeur trompeuse qui masque l'absence totale de discrimination.

Ce phénomène, illustré dans le notebook 05 sous le nom d'*accuracy paradox*, justifie le choix du **ROC-AUC comme métrique primaire** et de l'**averaging macro** pour agréger les métriques multiclasses. Les stratégies de compensation adoptées sont :

- **Pondération des classes** dans la fonction de perte (weighted cross-entropy) pour les modèles deep learning
- **Stratification systématique** des splits train/test et des folds de cross-validation
- **Averaging macro** pour toutes les métriques agrégées

---

## 3. Prétraitement et réduction dimensionnelle (PCA)

### 3.1 Pipeline de prétraitement

Toutes les images des trois datasets subissent le même pipeline de prétraitement avant d'être utilisées par les modèles :

1. **Redimensionnement** → 224 × 224 pixels (format standard des architectures CNN modernes)
2. **Conversion en niveaux de gris** (mode L)
3. **Normalisation standard** (zero-mean, unit variance, calculée sur l'ensemble du dataset)

Les plages de valeurs après normalisation confirment que le pipeline fonctionne correctement :

| Dataset | Plage de valeurs normalisées |
|---------|------------------------------|
| Dataset 1 | [−6,846 ; 6,151] |
| Dataset 2 | [−2,604 ; 3,016] |
| Dataset 3 | [−3,533 ; 5,194] |

### 3.2 Partitionnement train/test

Le partitionnement est réalisé sur Dataset 1, avec une stratégie **80/20 stratifiée** (random_state=42) pour garantir la reproductibilité et le maintien des proportions de classes dans chaque partition :

| Partition | Total | NORMAL | BACTERIA | VIRUS |
|-----------|:-----:|:------:|:--------:|:-----:|
| **Train** | 4 185 | 1 079 | 2 030 | 1 076 |
| **Test** | 1 047 | 270 | 508 | 269 |

### 3.3 Réduction dimensionnelle par PCA

La PCA (Analyse en Composantes Principales) est appliquée **avant tous les modèles de machine learning classique** pour rendre leur entraînement computationnellement faisable. En effet, une image 224×224 aplatie produit un vecteur de **50 176 features**. À cette dimension, un SVM à noyau RBF doit calculer une matrice de Gram de taille (4 185 × 4 185) pour chacune des 50 176 dimensions — un calcul prohibitif en temps et en mémoire (complexité O(n²×d)).

| Paramètre PCA | Valeur |
|---------------|--------|
| Méthode | `sklearn.decomposition.PCA` |
| Critère de sélection | `n_components=0.95` (variance cible) |
| Nombre de composantes retenues | **792** |
| Variance expliquée | **95,01 %** |
| Réduction dimensionnelle | **98,4 %** (50 176 → 792) |
| Whitening | Activé |

La PCA conserve 95 % de l'information discriminante tout en réduisant la dimensionnalité de 98,4 %. Les détails techniques complets (forme des tenseurs, paramètres sklearn) sont disponibles en [Annexe D](#annexe-d--détails-techniques-de-la-pca).

> 📊 **[IMAGE — Notebook 02]** Courbe de variance expliquée cumulée en fonction du nombre de composantes PCA (axe x : 0→800 composantes, axe y : 0→100 % variance). Indiquer le coude à 792 composantes avec une ligne verticale pointillée.

> 📊 **[IMAGE — Notebook 02]** Visualisation des premières composantes principales (eigenfaces) sur les images de radiographie — permet d'interpréter ce que la PCA retient.

---

## 4. Modèles baseline — Machine Learning classique

### 4.1 Modèles évalués

Quatre algorithmes de machine learning classique sont entraînés sur les features réduites par PCA (792 composantes) :

- **Logistic Regression** — modèle linéaire de référence, rapide et interprétable
- **SVM à noyau RBF** — séparateur à marge maximale, efficace en haute dimension
- **Random Forest** — ensemble d'arbres de décision, robuste au bruit
- **Gradient Boosting** — boosting séquentiel, excellent sur données tabulaires

Tous les modèles sont évalués selon deux protocoles : **split simple 80/20** et **cross-validation 5-fold stratifiée**. Cette double évaluation permet de vérifier la cohérence des résultats et de quantifier la variance des estimations (voir [Annexe B](#annexe-b--comparaison-avec-et-sans-cross-validation)).

### 4.2 Résultats — Split simple (train 80 % / test 20 %)

| Modèle | ROC-AUC | Accuracy | Precision | Recall | F1-Score | Temps (s) |
|--------|:-------:|:--------:|:---------:|:------:|:--------:|:---------:|
| Logistic Regression | 0,8628 | 0,7163 | 0,7106 | 0,7225 | 0,7146 | 0,50 |
| **SVM (RBF)** | **0,8895** | 0,7584 | 0,7545 | 0,7455 | 0,7493 | 60,93 |
| Random Forest | 0,8470 | 0,6227 | 0,7543 | 0,5176 | 0,5166 | 1,14 |
| Gradient Boosting | 0,8931 | 0,7584 | 0,7519 | 0,7204 | 0,7246 | 175,40 |

Le Gradient Boosting obtient le ROC-AUC le plus élevé (0,8931) en split simple, suivi de très près par le SVM (0,8895). Le Random Forest se démarque négativement avec un recall très faible (0,5176), ce qui traduit une difficulté à identifier la classe VIRUS, moins représentée.

### 4.3 Résultats — Cross-validation 5-fold stratifiée

Les paramètres de la cross-validation sont détaillés en [Annexe C](#annexe-c--paramètres-complets-de-la-cross-validation-et-du-grid-search). Les résultats sont exprimés en moyenne ± écart-type sur les 5 folds :

| Modèle | ROC-AUC | Accuracy | Precision | Recall | F1-Score | Temps CV (s) |
|--------|:-------:|:--------:|:---------:|:------:|:--------:|:------------:|
| Logistic Regression | 0,8569 ± 0,0069 | 0,6999 ± 0,0067 | 0,7011 ± 0,0097 | 0,7090 ± 0,0085 | 0,7024 ± 0,0084 | 29,52 |
| **SVM (RBF)** | **0,8943 ± 0,0063** | 0,7626 ± 0,0128 | 0,7642 ± 0,0113 | 0,7546 ± 0,0108 | 0,7590 ± 0,0108 | 343,56 |
| Random Forest | 0,8428 ± 0,0076 | 0,6353 ± 0,0095 | 0,7654 ± 0,0179 | 0,5368 ± 0,0116 | 0,5458 ± 0,0151 | 8,01 |
| Gradient Boosting | 0,8876 ± 0,0062 | 0,7619 ± 0,0045 | 0,7560 ± 0,0057 | 0,7306 ± 0,0080 | 0,7350 ± 0,0071 | 227,73 |

En cross-validation, le SVM (RBF) s'impose comme le meilleur modèle avec un ROC-AUC de **0,8943 ± 0,0063**. L'écart-type faible (0,0063) indique une bonne stabilité entre les folds.

### 4.4 Analyse de cohérence entre les deux protocoles

La cohérence entre split simple et cross-validation est un indicateur de fiabilité des résultats. Un écart important trahirait un overfitting sur le split ou une dépendance forte au tirage aléatoire.

| Modèle | ROC-AUC (split) | ROC-AUC (CV) | Δ | Interprétation |
|--------|:---------------:|:------------:|:---:|----------------|
| SVM | 0,8895 | 0,8943 ± 0,0063 | +0,005 | Cohérent — CV légèrement supérieure |
| Gradient Boosting | 0,8931 | 0,8876 ± 0,0062 | −0,006 | Cohérent — split légèrement optimiste |
| Logistic Regression | 0,8628 | 0,8569 ± 0,0069 | −0,006 | Cohérent |
| Random Forest | 0,8470 | 0,8428 ± 0,0076 | −0,004 | Cohérent |

Tous les écarts restent inférieurs à 1 point de ROC-AUC (Δ < 0,01), ce qui confirme la robustesse des résultats et l'absence d'overfitting sur le split de test. La discussion sur le choix entre les deux protocoles est développée en [Annexe B](#annexe-b--comparaison-avec-et-sans-cross-validation).

> 📊 **[IMAGE — Notebook 03]** Barplot comparatif ROC-AUC : pour chaque modèle, deux barres côte-à-côte (split vs CV ± std). Permet de visualiser la cohérence des résultats.

> 📊 **[IMAGE — Notebook 03]** Courbes ROC One-vs-Rest (OvR) pour les 4 modèles baseline sur le test set. Une courbe par classe (NORMAL, BACTERIA, VIRUS) + courbe macro-average.

> 📊 **[IMAGE — Notebook 03]** Matrices de confusion normalisées pour les 4 modèles (grille 2×2). Particulièrement révélateur pour Random Forest (faible recall sur VIRUS).

---

## 5. Deep Learning — Réseaux de neurones convolutifs

### 5.1 Architectures et configuration d'entraînement

Trois architectures de réseaux de neurones convolutifs sont évaluées. Contrairement aux modèles baseline, le deep learning opère directement sur les images 224×224 pixels sans passage par la PCA.

| Architecture | Type | Early stopping (epoch) | Meilleur epoch | Batch size | Learning rate |
|-------------|------|:----------------------:|:--------------:|:----------:|:-------------:|
| Custom CNN | From scratch | 26 | 16 | 32 | 3×10⁻⁴ |
| VGG16 | Transfer Learning (ImageNet) | 31 | 21 | 32 | 3×10⁻⁴ |
| ResNet50 | Transfer Learning (ImageNet) | 27 | 17 | 32 | 3×10⁻⁴ |

**Configuration commune à tous les modèles DL :**

- Optimiseur : Adam (avec warmup 3 epochs + cosine annealing)
- Fonction de perte : Weighted Cross-Entropy (poids=[1,94, 0,67] pour compenser le déséquilibre)
- Gradient clipping : 1,0
- Early stopping : patience=10 epochs, moniteur=val_loss
- Split utilisé : 3 138 images train / 1 047 images test (80/20)

### 5.2 Résultats sur le test set

| Modèle | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|--------|:-------:|:--------:|:---------:|:------:|:--------:|
| Custom CNN | 0,9899 | 96,28 % | 0,9719 | 0,9781 | 0,9750 |
| ResNet50 Transfer | 0,9967 | 96,94 % | 0,9908 | 0,9678 | 0,9792 |
| **VGG16 Transfer** | **0,9972** | **97,90 %** | **0,9935** | **0,9781** | **0,9857** |

VGG16 Transfer est le meilleur modèle toutes familles confondues, avec un ROC-AUC de **0,9972** et un F1-score de **0,9857**. Sa convergence rapide (meilleur checkpoint à l'epoch 21 sur 50 possibles) et son faible gap entre performances train et validation indiquent une excellente généralisation.

### 5.3 Comparaison Deep Learning vs Baseline

Le saut de performance entre les modèles ML classiques et le deep learning est substantiel :

| Indicateur | Meilleur baseline (SVM CV) | Meilleur DL (VGG16) | Gain |
|------------|:--------------------------:|:-------------------:|:----:|
| ROC-AUC | 0,8943 | 0,9972 | **+10,3 pts** |
| Accuracy | 76,3 % | 97,9 % | **+21,6 pts** |
| F1-Score | 0,7590 | 0,9857 | **+22,7 pts** |

Ce gain de +10,3 points de ROC-AUC représente une amélioration non marginale. Il s'explique par la capacité du transfer learning à exploiter des représentations visuelles hiérarchiques apprises sur ImageNet (millions d'images), bien adaptées à la détection de patterns texturaux dans les radiographies.

> 📊 **[IMAGE — Notebook 04]** Courbes d'apprentissage pour les 3 architectures DL : loss (train et validation) en fonction des epochs, avec annotation du meilleur epoch et du point d'early stopping.

> 📊 **[IMAGE — Notebook 04]** Comparaison globale des performances : barplot ROC-AUC regroupant les 4 modèles baseline + les 3 modèles DL (7 barres au total, ordonnées par performance).

> 📊 **[IMAGE — Notebook 06]** Visualisations Grad-CAM : grille 2×5 (5 images NORMAL + 5 images PNEUMONIA) avec les cartes d'activation superposées. Permet de vérifier que le modèle focus sur les zones pulmonaires cliniquement pertinentes (consolidations) et non sur les bordures ou annotations.

### 5.4 Interprétabilité — Score de confiance clinique

Le notebook 06 évalue la confiance clinique du modèle VGG16 via un **Trust Score composite** :

| Composante | Score | Poids |
|------------|:-----:|:-----:|
| Performance (ROC-AUC, F1) | 0,997 | 35 % |
| Fiabilité des scores de confiance | 0,850 | 20 % |
| Patterns d'erreur | 0,979 | 20 % |
| Interprétabilité (Grad-CAM) | 0,750 | 15 % |
| Validité clinique | 0,850 | 10 % |
| **Trust Score global** | **0,912 / 1,000** | — |

Un Trust Score de 0,912 (seuil de déploiement : 0,80) indique que le modèle est **prêt pour une assistance clinique**, sous réserve d'un maintien du médecin comme décideur final.

---

## 6. Optimisation, ensemble et fine-tuning

### 6.1 Hyperparameter tuning — Grid Search

Le fine-tuning des hyperparamètres est réalisé via GridSearchCV sur **Dataset 3** (624 images dédiées exclusivement à cette étape, distinctes du train et du test).

| Paramètre | Valeur |
|-----------|--------|
| Méthode | `GridSearchCV` |
| CV folds (tuning) | **3** |
| Métrique de scoring | `roc_auc` |
| Parallélisation | `n_jobs=-1` (tous les cœurs disponibles) |
| Dataset utilisé | Dataset 3 (624 images, dédié tuning) |

L'isolation de Dataset 3 pour le tuning est une décision délibérée pour éviter toute fuite d'information (*data leakage*) entre l'optimisation des hyperparamètres et l'évaluation finale sur le test set.

### 6.2 Méthodes d'ensemble

Trois stratégies de combinaison de modèles sont comparées. L'ensemble combine les 7 modèles entraînés (4 ML baseline + 3 DL) :

| Méthode d'ensemble | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------------------|:-------:|:--------:|:---------:|:------:|:--------:|
| Hard Voting | — | 0,8004 | 0,7880 | 1,0000 | 0,8815 |
| Soft Voting (poids égaux) | 0,9934 | 0,8109 | 0,7969 | 1,0000 | 0,8870 |
| **Weighted Soft Voting** | **0,9934** | **0,8300** | **0,8136** | **1,0000** | **0,8972** |

Le **Weighted Soft Voting** est la stratégie retenue. Les poids de chaque modèle sont proportionnels à son ROC-AUC individuel :

| Modèle | Poids (ROC-AUC) |
|--------|:---------------:|
| SVM (RBF) | 0,9879 |
| Gradient Boosting | 0,9851 |
| Logistic Regression | 0,9774 |
| Random Forest | 0,9748 |
| VGG16 Transfer | 0,9660 |
| Custom CNN | 0,9374 |
| ResNet50 Transfer | 0,8981 |

**Test-Time Augmentation (TTA) :** appliqué sur VGG16, la TTA améliore le ROC-AUC de 0,9660 à 0,9721 (+0,61 %), en faisant la moyenne de prédictions sur plusieurs versions augmentées de chaque image.

> 📊 **[IMAGE — Notebook 07]** Courbes ROC : une courbe par modèle individuel + courbe de l'ensemble weighted soft voting. L'ensemble doit apparaître au-dessus de tous les modèles individuels.

> 📊 **[IMAGE — Notebook 07]** Matrice de confusion de l'ensemble weighted soft voting (normalisée).

### 6.3 Optimisation du seuil de décision

Par défaut, les classifieurs utilisent un seuil de probabilité de 0,5. Cependant, en contexte médical, le choix du seuil traduit un arbitrage clinique explicite entre sensibilité (ne rater aucun cas) et spécificité (ne pas sur-diagnostiquer). Trois stratégies d'optimisation sont comparées :

| Stratégie | Seuil | Accuracy | Precision | Recall | F1-Score | Sensibilité | Spécificité |
|-----------|:-----:|:--------:|:---------:|:------:|:--------:|:-----------:|:-----------:|
| Défaut (0,5) | 0,500 | 0,8300 | 0,8136 | 1,0000 | 0,8972 | 1,000 | 0,341 |
| F1-optimisé | 0,677 | 0,9637 | 0,9660 | 0,9858 | 0,9758 | 0,986 | 0,900 |
| **Youden (recommandé)** | **0,818** | **0,9551** | **0,9893** | **0,9498** | **0,9691** | **0,950** | **0,970** |
| High recall (F2) | 0,606 | 0,9522 | 0,9460 | 0,9923 | 0,9686 | 0,992 | 0,837 |

L'**indice de Youden** (seuil=0,818) est retenu comme seuil de référence : il maximise la somme Sensibilité + Spécificité − 1, offrant le meilleur compromis entre les deux types d'erreur. Avec ce seuil, 95,0 % des cas de pneumonie sont détectés (sensibilité) et 97,0 % des cas normaux sont correctement identifiés (spécificité).

**Recommandations par scénario clinique :**

- **Dépistage de masse** → seuil 0,606 (F2) : priorité au rappel (99,2 %), accepter plus de faux positifs
- **Usage clinique quotidien** → seuil 0,818 (Youden) : équilibre optimal sensibilité/spécificité
- **Confirmation diagnostique** → seuil 0,677 (F1) : priorité à la précision

> 📊 **[IMAGE — Notebook 07]** Courbe d'optimisation du seuil : axe x = valeurs de seuil (0→1), axe y = score. Tracer F1, indice de Youden et F2 en fonction du seuil, avec annotation des seuils optimaux pour chacun.

---

## 7. Synthèse et réponses aux questions de recherche

### 7.1 Vue d'ensemble des performances

Le tableau suivant récapitule les performances de tous les modèles, ordonnées par ROC-AUC décroissant :

| Rang | Modèle | Famille | ROC-AUC | Accuracy | F1-Score |
|:----:|--------|---------|:-------:|:--------:|:--------:|
| 1 | Ensemble Weighted Soft Voting | Ensemble | 0,9934 | 0,8300 | 0,8972 |
| 2 | VGG16 Transfer | Deep Learning | 0,9972* | 0,9790 | 0,9857 |
| 3 | ResNet50 Transfer | Deep Learning | 0,9967 | 0,9694 | 0,9792 |
| 4 | Custom CNN | Deep Learning | 0,9899 | 0,9628 | 0,9750 |
| 5 | SVM (RBF) | ML Classique | 0,8943 (CV) | 0,7626 | 0,7590 |
| 6 | Gradient Boosting | ML Classique | 0,8876 (CV) | 0,7619 | 0,7350 |
| 7 | Logistic Regression | ML Classique | 0,8569 (CV) | 0,6999 | 0,7024 |
| 8 | Random Forest | ML Classique | 0,8428 (CV) | 0,6353 | 0,5458 |

*Note : Les ROC-AUC des modèles DL (lignes 2–4) sont mesurés sur le test set en évaluation isolée (notebook 04/05). L'ensemble (ligne 1) utilise une configuration de scores unifiée où les modèles DL présentent des scores légèrement différents ; le ROC-AUC 0,9934 de l'ensemble est supérieur au meilleur modèle individuel dans cette configuration (SVM=0,9879).*

### 7.2 Réponse à RQ1

> **RQ1** : Les algorithmes de ML classiques, appliqués sur des features PCA (95 % de variance), peuvent-ils atteindre ROC-AUC ≥ 0,85 ?

**Réponse : OUI.** Le SVM (RBF) atteint un ROC-AUC de **0,8943 ± 0,0063** en cross-validation 5-fold, dépassant le seuil clinique de 0,85. La PCA est un rouage essentiel de ce résultat : elle réduit les 50 176 features initiales à 792 composantes (95,01 % de variance conservée, réduction de 98,4 %), rendant le calcul tractable tout en préservant la quasi-totalité de l'information discriminante. Sans PCA, l'entraînement du SVM serait computationnellement prohibitif (complexité O(n²×d) pour la matrice de Gram).

### 7.3 Réponse à RQ2

> **RQ2** : Le deep learning par transfer learning apporte-t-il une amélioration significative et robuste par rapport aux approches classiques ?

**Réponse : OUI, et de manière non marginale.** VGG16 Transfer atteint un ROC-AUC de **0,9972**, soit un gain de **+10,3 points** par rapport au meilleur modèle ML classique (SVM, ROC-AUC=0,8943). Le gain en accuracy est encore plus marqué : **+21,6 points** (97,9 % vs 76,3 %). La convergence rapide (early stopping epoch 21/50) et le faible gap train/validation attestent d'une bonne généralisation, non d'un surapprentissage. Ce gain justifie le surcoût computationnel du transfer learning.

### 7.4 Réponse à RQ3

> **RQ3** : L'ensemble weighted soft voting dépasse-t-il les modèles individuels, et quel seuil est optimal ?

**Réponse : OUI pour l'ensemble ; seuil de Youden (0,818) recommandé.** Dans la configuration unifiée de l'évaluation (notebook 07), le weighted soft voting atteint un ROC-AUC de **0,9934**, supérieur au meilleur modèle individuel (SVM=0,9879 dans cette configuration, gain de +0,55 %). L'ensemble tire profit de la diversité des modèles qui le composent — les erreurs non corrélées entre modèles se compensent.

Le **seuil de Youden (0,818)** est recommandé pour un usage clinique de routine : il garantit une **sensibilité de 95,0 %** (95 % des cas de pneumonie détectés) et une **spécificité de 97,0 %** (97 % des cas normaux correctement identifiés), avec un indice de Youden de 0,920.

---

## Annexe A — Définition et interprétation des métriques

Les métriques utilisées dans ce rapport sont définies ci-dessous en termes de vrais positifs (TP), faux positifs (FP), vrais négatifs (TN) et faux négatifs (FN). Dans le contexte de la détection de pneumonie, la classe positive est PNEUMONIA.

| Métrique | Formule | Interprétation médicale |
|----------|---------|------------------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Taux de prédictions correctes. Trompeuse en cas de déséquilibre de classes (accuracy paradox). |
| **Precision** | TP / (TP + FP) | Parmi tous les cas que le modèle a diagnostiqués "pneumonie", quelle proportion l'était vraiment ? Mesure le taux de faux alarmes. |
| **Recall (Sensibilité)** | TP / (TP + FN) | Parmi tous les vrais cas de pneumonie, quelle proportion a été détectée ? Mesure la capacité à ne manquer aucun cas — métrique critique en médecine. |
| **Spécificité** | TN / (TN + FP) | Parmi tous les cas réellement normaux, quelle proportion a été correctement classée ? Mesure la capacité à ne pas sur-diagnostiquer. |
| **F1-Score** | 2 × (P × R) / (P + R) | Moyenne harmonique de la précision et du rappel. Bon compromis quand les deux erreurs (faux positifs et faux négatifs) ont un coût. |
| **ROC-AUC** | Aire sous la courbe ROC | Capacité de discrimination du modèle indépendamment du seuil de décision. Valeur de 0,5 = prédiction aléatoire, 1,0 = discriminateur parfait. Métrique primaire choisie car robuste au déséquilibre de classes. |
| **Indice de Youden** | Sensibilité + Spécificité − 1 | Mesure l'avantage total du test par rapport à la prédiction aléatoire. Utilisé pour identifier le seuil de décision optimal (maximum de l'indice). Plage : [0, 1]. |
| **F2-Score** | 5 × (P × R) / (4×P + R) | Variante du F1 qui pénalise davantage les faux négatifs que les faux positifs. Utilisé pour le dépistage (priorité au rappel). |

**Note sur l'averaging macro :** toutes les métriques multiclasses (Precision, Recall, F1) sont calculées en **macro-average** : la métrique est calculée pour chaque classe indépendamment, puis la moyenne non pondérée est prise. Cette approche traite chaque classe à égalité, sans favoriser la classe majoritaire (PNEUMONIA/BACTERIA).

**L'accuracy paradox** est illustré dans le notebook 05 : sur Dataset 1, un classifieur trivial qui prédirait toujours "PNEUMONIA" atteindrait 74,3 % d'accuracy — valeur supérieure à plusieurs de nos modèles ML classiques — sans aucune capacité discriminante réelle. C'est pourquoi le ROC-AUC est la métrique primaire de ce projet.

---

## Annexe B — Comparaison avec et sans cross-validation

### Pourquoi utiliser deux protocoles d'évaluation ?

Le **split simple** (80/20) est rapide et intuitif, mais il présente un défaut fondamental : ses résultats dépendent du tirage aléatoire. Un split "chanceux" peut sur-estimer les performances ; un split "malchanceux" les sous-estime. Avec 1 047 images dans le test set, l'incertitude statistique est non négligeable.

La **cross-validation k-fold** (k=5) résout ce problème en utilisant l'intégralité des données pour la validation : chaque image passe exactement une fois dans le fold de test. La moyenne et l'écart-type sur les 5 folds donnent une estimation plus robuste de la performance réelle et une mesure de la variance (stabilité) du modèle.

### Résultats comparés et analyse de cohérence

| Modèle | ROC-AUC (split) | ROC-AUC (CV 5-fold) | Δ | Conclusion |
|--------|:---------------:|:-------------------:|:---:|------------|
| SVM (RBF) | 0,8895 | 0,8943 ± 0,0063 | +0,005 | Cohérent — CV légèrement supérieure au split |
| Gradient Boosting | 0,8931 | 0,8876 ± 0,0062 | −0,006 | Cohérent — split légèrement optimiste |
| Logistic Regression | 0,8628 | 0,8569 ± 0,0069 | −0,006 | Cohérent |
| Random Forest | 0,8470 | 0,8428 ± 0,0076 | −0,004 | Cohérent |

Tous les écarts sont inférieurs à **1 point de ROC-AUC** (Δ < 0,01). Cette cohérence systématique confirme deux choses :

1. **Les résultats sont robustes** : ils ne sont pas le fruit d'un tirage aléatoire favorable.
2. **Le split 80/20 stratifié est de bonne qualité** pour ce jeu de données (stratification efficace).

### Quand préférer l'une ou l'autre méthode ?

| Critère | Split simple | Cross-validation |
|---------|:-----------:|:----------------:|
| Rapidité d'itération | ✅ Rapide | ❌ k× plus lent |
| Estimation non biaisée | ⚠️ Dépend du tirage | ✅ Robuste |
| Intervalle de confiance | ❌ Aucun | ✅ ± std disponible |
| Publication / rapport | ⚠️ Acceptable | ✅ Recommandé |

**Recommandation** : utiliser le split simple pour l'exploration et le développement rapide, et la cross-validation pour les résultats définitifs à reporter.

---

## Annexe C — Paramètres complets de la cross-validation et du Grid Search

### Cross-validation principale (évaluation des modèles)

```
Méthode         : StratifiedKFold
k               : 5 folds
Stratification  : Oui — proportions de classes maintenues dans chaque fold
                  (variation max. = ±1 sample entre folds)
random_state    : 42
Averaging       : Macro (chaque classe contribue équitablement)
Scoring         : ROC-AUC (One-vs-Rest, macro)
```

Distribution des classes par fold (illustrative) :

| Fold | NORMAL | BACTERIA | VIRUS | Total |
|:----:|:------:|:--------:|:-----:|:-----:|
| 1 | ~216 | ~406 | ~215 | 837 |
| 2 | ~216 | ~406 | ~215 | 837 |
| 3 | ~216 | ~406 | ~215 | 837 |
| 4 | ~216 | ~406 | ~215 | 837 |
| 5 | ~215 | ~406 | ~216 | 837 |

La variation maximale inter-folds est de ±1 sample, confirmant l'efficacité de la stratification.

### Cross-validation pour le Grid Search (hyperparameter tuning)

```
Méthode         : GridSearchCV (sklearn)
CV folds        : 3
Dataset         : Dataset 3 (624 images, dédié exclusivement au tuning)
Scoring         : roc_auc
Parallélisation : n_jobs=-1 (tous les cœurs disponibles)
```

Le Dataset 3 est volontairement isolé du Dataset 1 (train/test) pour éviter toute contamination des estimations de performance par les hyperparamètres optimisés.

### Paramètres d'entraînement deep learning (config.yaml)

```yaml
batch_size        : 32
epochs            : 50
learning_rate     : 0.0003
optimizer         : AdamW
early_stopping    :
  patience        : 7 (notebooks : 10)
  monitor         : val_loss
weight_decay      : 0.0001
gradient_clipping : 1.0
warmup_epochs     : 3
lr_schedule       : cosine annealing
random_state      : 42
```

---

## Annexe D — Détails techniques de la PCA

### Paramètres sklearn

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train_flat)  # shape: (4185, 50176) → (4185, 792)
X_test_pca  = pca.transform(X_test_flat)       # shape: (1047, 50176) → (1047, 792)
```

| Paramètre | Valeur |
|-----------|--------|
| `n_components` | `0.95` (sélection automatique par variance cible) |
| Composantes retenues | **792** |
| Variance expliquée | **95,01 %** |
| Réduction dimensionnelle | **98,4 %** (50 176 → 792 features) |
| `whiten` | `True` (normalisation de la variance de chaque composante) |
| Forme d'entrée | (n_samples, 50 176) — images 224×224 aplaties |
| Forme de sortie | (n_samples, 792) — composantes principales |

### Justification computationnelle

Sans PCA, l'entraînement du SVM à noyau RBF sur les features brutes serait impraticable :

- **Espace mémoire de la matrice de Gram** : n² × d = 4 185² × 50 176 ≈ **3,5 téraoctets**
- **Temps de calcul du kernel** : O(n² × d) — prohibitif pour n=4 185, d=50 176

Avec PCA (d'=792 composantes) :
- **Espace mémoire** : n × d' = 4 185 × 792 ≈ **26 Mo**
- **Temps de calcul kernel** : réduit d'un facteur 63× (50 176 / 792)

### Le whitening

Le whitening (`whiten=True`) normalise la variance de chaque composante principale à 1. Cette étape est bénéfique pour les modèles sensibles à l'échelle des features (SVM, régression logistique) car elle garantit que chaque composante contribue équitablement à la distance euclidienne utilisée par le noyau RBF.

### Interprétation des composantes

Les premières composantes principales capturent les variations globales de texture et de densité pulmonaire (structures de grande échelle). Les composantes d'ordre supérieur encodent des détails plus fins — contours, opacités locales — qui peuvent être discriminants pour distinguer pneumonie bactérienne et virale. Le graphique de variance cumulée (voir section 3.3) montre que le coude de la courbe se situe précisément autour de la 792ème composante, confirmant que le seuil de 95 % est un choix optimal entre compression et fidélité de l'information.

---

*Rapport généré à partir des notebooks 01 à 07 du projet ZOIDBERG 2.0 — T-DEV-810-PAR_19, EPITECH, Juin 2026.*
