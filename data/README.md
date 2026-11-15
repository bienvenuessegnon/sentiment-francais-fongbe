# Données

Instantané constitué le 15/11/2026. Les fichiers volumineux restent dans
`data/external/` (non versionnés) et se retéléchargent au besoin. Les jeux
utilisés pour l'entraînement et l'évaluation sont dans `data/processed/`.

## Sources

### 1. Allociné — critiques de films en français (étiquettes d'origine)

| | |
| --- | --- |
| **Jeu** | Allociné dataset (Théophile Blard, 2020) |
| **Dépôt** | <https://huggingface.co/datasets/tblard/allocine> |
| **Code d'origine** | <https://github.com/TheophileBlard/french-sentiment-analysis-with-bert> |
| **Licence** | MIT |
| **Contenu** | 160 000 / 20 000 / 20 000 critiques (train / validation / test), étiquette binaire issue de la note de l'utilisateur (négatif = 0, positif = 1) |
| **Usage ici** | sous-échantillons équilibrés de 10 000 / 2 000 / 2 000 textes, longueur 40–700 caractères, **découpage officiel conservé** |

Aucune critique du test officiel n'entre dans l'apprentissage.

### 2. Corpus parallèle Fon–Français (CC BY 4.0)

Il n'existe pas, à notre connaissance, de corpus de sentiment fongbé annoté
par des locuteurs natifs. AfriSenti (Muhammad et al., 2023) couvre 14 langues
africaines, **sans le fon**. Le fongbé dispose en revanche de corpus parallèles
ouverts, dont on se sert pour transférer une polarité lue en français.

Le fichier utilisé est l'édition consolidée, normalisée et dédupliquée :

- <https://huggingface.co/datasets/Flex1-tech/fon-french-parallel-corpus> (57 458 paires uniques)

Elle rassemble deux sources primaires, toutes deux en **CC BY 4.0** :

1. Dossou, Yoclounon, Ahounvlamè et Emezue, 2021, *Fon French Daily Dialogues Parallel Data*, [Zenodo 4432712](https://doi.org/10.5281/zenodo.4432712)
2. Degila, Kalipe, Touré Ali et Balogoun, 2020, *Parallel text dataset for Neural Machine Translation (French → Fongbe, French → Ewe)*, [Zenodo 4266935](https://doi.org/10.5281/zenodo.4266935)

Le dépôt historique FFR v1 (<https://github.com/bonaventuredossou/ffr-v1>,
aussi sur <https://huggingface.co/datasets/FrancophonIA/FFR>) n'est pas utilisé :
sa licence est plus restrictive (CC BY-NC-SA et mention d'un accord des auteurs
pour un usage commercial).

### 3. Corpus « Fon Sentiment » (Owusu, 2025) — écarté

| | |
| --- | --- |
| **Jeu** | [michsethowusu/fon-sentiments-corpus](https://huggingface.co/datasets/michsethowusu/fon-sentiments-corpus) |
| **Licence** | MIT |
| **Étiquettes** | automatiques : DistilBERT anglais appliqué à une traduction, d'après la fiche du jeu |
| **Audit (15/11/2026)** | 34 915 lignes ; seulement 42 % contiennent une lettre typique du fongbé (ɔ, ɛ, ɖ) ; 16 791 lignes en alphabet latin sans ces lettres ; 4 559 ont une allure clairement française (voir `data/raw/audit_fon_sentiments.json`) |

Ce jeu n'entre ni dans l'apprentissage ni dans l'évaluation : les étiquettes
ne sont pas humaines, et le mélange de langues y est visible.

## Projection d'étiquettes vers le fongbé

1. On garde les paires assez longues (au moins 4 mots côté français, 3 côté fongbé, 20 à 220 caractères en français).
2. Un **lexique français conservative**, listé dans `src/lexique.py` (ce n'est pas un lexique tiers), attribue une polarité seulement s'il trouve des indices d'un seul côté, avec des frontières de mots (« beau » ne match pas « beaucoup »).
3. Les phrases avec négation (`ne … pas`, `sans peur`, etc.) ou une locution figée (`nulle part`, `beaux-parents`, `de peur que`, `avoir beau`) sont écartées, pas inversées.
4. L'étiquette retenue est transférée au texte fongbé.
5. Les partitions se font **par groupe** (squelette lexical, casse et diacritiques retirés) pour limiter la fuite des quasi-doublons, comme le recommande la notice du corpus consolidé.

Un classifieur entraîné sur Allociné a été essayé pour cette projection : sur
des phrases courtes de dialogue, il classe presque tout en « négatif » (décalage
de domaine). Il n'est donc pas utilisé pour étiqueter le fongbé.

Résultat après filets : 1 036 paires projetées (668 positives, 368 négatives),
puis 701 / 115 / 220 en train / val / test.

## Jeu fongbé vérifié

`data/processed/fon_verifie.csv` (68 phrases, 36 positives et 32 négatives)
est un sous-ensemble relu **sur le français** : la polarité du français a été
confirmée à la main, l'étiquette du fongbé reste transférée. Ces groupes sont
entièrement hors apprentissage, validation et test projetés. Ce n'est pas une
annotation par un locuteur natif du fongbé.

## Identification de langue

`lid_train.csv` et `lid_test.csv` mélangent le français et le fongbé des paires
parallèles (textes distincts d'un côté et de l'autre) plus un peu de français
Allociné. L'orthographe du fongbé (ɔ, ɛ, ɖ, tons) rend la tâche facile ; les
scores très élevés étaient attendus.

## Attribution

```
Blard, Théophile. French sentiment analysis with BERT, 2020.
https://github.com/TheophileBlard/french-sentiment-analysis-with-bert

Dossou, Bonaventure F. P. ; Yoclounon, Fabroni ; Ahounvlamè, Ricardo ; Emezue, Chris C.
Fon French Daily Dialogues Parallel Data, Zenodo, 2021. https://doi.org/10.5281/zenodo.4432712

Degila, Kevin ; Kalipe, Godson ; Touré Ali, Jamiil ; Balogoun, Momboladji.
Parallel text dataset for Neural Machine Translation (French -> Fongbe, French -> Ewe),
Zenodo, 2020. https://doi.org/10.5281/zenodo.4266935
```
