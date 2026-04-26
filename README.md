# heart-failure-prediction-logistic-regression

## Description :
Ce projet vise à prédire le risque de décès chez des patients atteints d’insuffisance cardiaque en utilisant un modèle de régression logistique (GLM).

## Dataset :
Heart Failure Clinical Records Dataset (299 patients)
Méthodologie

## Analyse exploratoire des données
Split stratifié (80% train / 20% test)
Modélisation GLM (logit)
Sélection de variables (Stepwise AIC)
Validation avec matrice de confusion et ROC

## Résultats
Accuracy test : ~76%
AUC test : ~0.79
Modèle final avec 5 variables clés :
age
ejection_fraction
serum_creatinine
serum_sodium
time

