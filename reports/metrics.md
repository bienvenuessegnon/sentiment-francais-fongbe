# Métriques

Généré le 2026-11-15. Tous les chiffres viennent du code.

## Tableau récapitulatif

| modele | jeu | f1_macro | accuracy |
| --- | --- | --- | --- |
| TF-IDF fr | fr_test | 0.912 | 0.912 |
| TF-IDF fr | fon_proj_test | 0.378 | 0.427 |
| TF-IDF fr | fon_verifie | 0.398 | 0.500 |
| TF-IDF fr | fr_val | 0.915 | 0.915 |
| TF-IDF fr+fon | fr_test | 0.916 | 0.916 |
| TF-IDF fr+fon | fon_proj_test | 0.816 | 0.827 |
| TF-IDF fr+fon | fon_verifie | 0.807 | 0.809 |
| TF-IDF fr+fon | fr_val | 0.911 | 0.911 |

## Points clés

- **F1 macro TF-IDF français sur le test Allociné** : 0.912
- **F1 macro TF-IDF français appliqué au fongbé projeté (zéro-shot lexical)** : 0.378
- **F1 macro TF-IDF fr+fon sur le test fongbé projeté** : 0.816
- **F1 macro TF-IDF fr+fon sur le jeu fongbé vérifié** : 0.807
- **Identification fr/fon (F1 macro)** : 0.997
- **Taille apprentissage français** : 10000
- **Taille apprentissage fongbé projeté** : 701
- **Taille jeu fongbé vérifié** : 68

## Détail des rapports de classification

### TF-IDF fr / fr_test

```
precision    recall  f1-score   support

     négatif       0.91      0.92      0.91      1000
     positif       0.92      0.91      0.91      1000

    accuracy                           0.91      2000
   macro avg       0.91      0.91      0.91      2000
weighted avg       0.91      0.91      0.91      2000
```

### TF-IDF fr / fon_proj_test

```
precision    recall  f1-score   support

     négatif       0.40      0.90      0.55        87
     positif       0.64      0.12      0.20       133

    accuracy                           0.43       220
   macro avg       0.52      0.51      0.38       220
weighted avg       0.55      0.43      0.34       220
```

### TF-IDF fr / fon_verifie

```
precision    recall  f1-score   support

     négatif       0.48      0.97      0.65        32
     positif       0.75      0.08      0.15        36

    accuracy                           0.50        68
   macro avg       0.62      0.53      0.40        68
weighted avg       0.62      0.50      0.38        68
```

### TF-IDF fr / fr_val

```
precision    recall  f1-score   support

     négatif       0.92      0.91      0.91      1000
     positif       0.91      0.92      0.92      1000

    accuracy                           0.92      2000
   macro avg       0.92      0.92      0.92      2000
weighted avg       0.92      0.92      0.92      2000
```

### TF-IDF fr+fon / fr_test

```
precision    recall  f1-score   support

     négatif       0.91      0.92      0.92      1000
     positif       0.92      0.91      0.92      1000

    accuracy                           0.92      2000
   macro avg       0.92      0.92      0.92      2000
weighted avg       0.92      0.92      0.92      2000
```

### TF-IDF fr+fon / fon_proj_test

```
precision    recall  f1-score   support

     négatif       0.81      0.74      0.77        87
     positif       0.84      0.89      0.86       133

    accuracy                           0.83       220
   macro avg       0.82      0.81      0.82       220
weighted avg       0.83      0.83      0.83       220
```

### TF-IDF fr+fon / fon_verifie

```
precision    recall  f1-score   support

     négatif       0.83      0.75      0.79        32
     positif       0.79      0.86      0.83        36

    accuracy                           0.81        68
   macro avg       0.81      0.81      0.81        68
weighted avg       0.81      0.81      0.81        68
```

### TF-IDF fr+fon / fr_val

```
precision    recall  f1-score   support

     négatif       0.92      0.90      0.91      1000
     positif       0.90      0.92      0.91      1000

    accuracy                           0.91      2000
   macro avg       0.91      0.91      0.91      2000
weighted avg       0.91      0.91      0.91      2000
```
