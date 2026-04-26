#   REGRESSION LOGISTIQUE SOUS R
#   Prediction du Deces par Insuffisance Cardiaque
#   Dataset : Heart Failure Clinical Records (Kaggle, 299 obs.)
#   Variable reponse : DEATH_EVENT (1 = Decede, 0 = Survivant)
# ================================================================
# ---------------------------------------------------------------
# ETAPE 0 : PACKAGES
# ---------------------------------------------------------------

install.packages(c("ggplot2", "dplyr", "MASS", "car",
                   "pROC", "lmtest", "corrplot", "caret"))

library(ggplot2); library(dplyr) ; library(MASS) ; library(car) library(pROC) ; library(lmtest) ; library(corrplot) ; library(caret)

# ---------------------------------------------------------------
# ETAPE 1 : CHARGEMENT ET PREPARATION DES DONNEES
# ---------------------------------------------------------------

df <- read.csv("heart_failure_clinical_records_dataset.csv")

str(df)
cat("Observations :", nrow(df), "\n")
cat("Variables    :", ncol(df), "\n")

print(colSums(is.na(df)))
cat("Nombre de lignes dupliquees :", sum(duplicated(df)), "\n")

print(summary(df))

tab_rep <- table(df$DEATH_EVENT)
print(tab_rep)
print(round(prop.table(tab_rep) * 100, 1))

df$anaemia             <- as.factor(df$anaemia)
df$diabetes            <- as.factor(df$diabetes)
df$high_blood_pressure <- as.factor(df$high_blood_pressure)
df$sex                 <- as.factor(df$sex)
df$smoking             <- as.factor(df$smoking)
df$DEATH_EVENT         <- as.factor(df$DEATH_EVENT)

# ---------------------------------------------------------------
# ETAPE 2 : ANALYSE EXPLORATOIRE
# ---------------------------------------------------------------

vars_cont <- c("age", "creatinine_phosphokinase", "ejection_fraction",
               "platelets", "serum_creatinine", "serum_sodium", "time")

# --- Boxplots ---
par(mfrow = c(2, 4), mar = c(5, 4, 3, 1))
for (v in vars_cont) {
  boxplot(as.numeric(df[[v]]) ~ df$DEATH_EVENT,
          main   = v,
          names  = c("Survie", "Deces"),
          col    = c("#4E79A7", "#E15759"),
          border = "gray40",
          ylab   = "")
}
par(mfrow = c(1, 1))

# --- Matrice de correlation ---
df_num <- df
df_num[vars_cont] <- lapply(df_num[vars_cont], as.numeric)
corrplot(cor(df_num[, vars_cont]),
         method      = "color",
         type        = "upper",
         tl.cex      = 0.8,
         col         = colorRampPalette(c("#E15759", "white", "#4E79A7"))(200),
         addCoef.col = "black",
         number.cex  = 0.7,
         tl.col      = "black",
         title       = "Matrice de Correlation",
         mar         = c(0, 0, 2, 0))
         
# ---------------------------------------------------------------
# ETAPE 3 : SPLIT 80/20 STRATIFIE
# ---------------------------------------------------------------

set.seed(42)

index_train <- createDataPartition(
  y    = df$DEATH_EVENT,
  p    = 0.80,
  list = FALSE
)
train <- df[ index_train, ]
test  <- df[-index_train, ]

cat("Train :", nrow(train), "observations\n")
cat("Test  :", nrow(test),  "observations\n")

print(round(prop.table(table(train$DEATH_EVENT)) * 100, 1))
print(round(prop.table(table(test$DEATH_EVENT))  * 100, 1))

train$DEATH_EVENT_num <- as.numeric(train$DEATH_EVENT) - 1
test$DEATH_EVENT_num  <- as.numeric(test$DEATH_EVENT)  - 1
# ---------------------------------------------------------------
# ETAPE 4 : MODELES LOGISTIQUES
# ---------------------------------------------------------------

# --- Modele nul ---
model_nul <- glm(DEATH_EVENT_num ~ 1,
                 family = binomial(link = "logit"),
                 data   = train)

# --- Modele complet ---
model_complet <- glm(
  DEATH_EVENT_num ~ age + anaemia + creatinine_phosphokinase +
    diabetes + ejection_fraction + high_blood_pressure +
    platelets + serum_creatinine + serum_sodium + sex + smoking + time,
  family = binomial(link = "logit"),
  data   = train
)
print(summary(model_complet))

cat("Deviance nulle     :", round(model_complet$null.deviance, 3), "\n")
cat("Deviance residuelle:", round(deviance(model_complet), 3), "\n")
cat("AIC                :", round(AIC(model_complet), 2), "\n")
cat("BIC                :", round(BIC(model_complet), 2), "\n")

# --- Odds Ratios modele complet ---
OR_complet <- exp(cbind(OR    = coef(model_complet),
                        IC_inf = confint.default(model_complet)[, 1],
                        IC_sup = confint.default(model_complet)[, 2]))
print(round(OR_complet, 4))

# --- Modele final (Stepwise AIC) ---
model_final <- stepAIC(model_complet, direction = "both", trace = FALSE)

print(summary(model_final))
cat("Variables retenues :", paste(names(coef(model_final))[-1], collapse = ", "), "\n")
cat("AIC modele final   :", round(AIC(model_final), 2), "\n")
cat("BIC modele final   :", round(BIC(model_final), 2), "\n")

# --- Odds Ratios modele final ---
OR_final <- exp(cbind(OR    = coef(model_final),
                      IC_inf = confint.default(model_final)[, 1],
                      IC_sup = confint.default(model_final)[, 2]))
print(round(OR_final, 4))

# ---------------------------------------------------------------
# ETAPE 5 : TESTS D'HYPOTHESES
# ---------------------------------------------------------------
# --- Test LRT : modele final vs modele nul ---
print(lrtest(model_nul, model_final))

# --- Test LRT : modele complet vs modele final ---
print(lrtest(model_final, model_complet))

# --- Test de Wald ---
print(round(coef(summary(model_final)), 4))

# --- Pseudo R2 ---
L0 <- as.numeric(logLik(model_nul))
LM <- as.numeric(logLik(model_final))
n  <- nrow(train)

r2_mcf <- 1 - LM / L0
r2_cox <- 1 - exp((2 / n) * (L0 - LM))
r2_nag <- r2_cox / (1 - exp(2 * L0 / n))

cat("McFadden  :", round(r2_mcf, 4), "\n")
cat("Cox-Snell :", round(r2_cox, 4), "\n")
cat("Nagelkerke:", round(r2_nag, 4), "\n")

# ---------------------------------------------------------------
# ETAPE 6 : ADEQUATION DU MODELE
# ---------------------------------------------------------------
# --- Test par la deviance ---
dev_res <- deviance(model_final)
ddl     <- df.residual(model_final)
p_dev   <- 1 - pchisq(dev_res, ddl)

cat("Deviance residuelle :", round(dev_res, 4), "\n")
cat("Degres de liberte   :", ddl, "\n")
cat("p-value (chi2)      :", round(p_dev, 4), "\n")
if (p_dev > 0.05) cat("=> Bonne adequation du modele (p > 0.05)\n") else
  cat("=> Manque d adequation (p < 0.05)\n")

# --- Test de Hosmer-Lemeshow ---
hoslem_test <- function(y, yhat, g = 10) {
  coupe    <- quantile(yhat, probs = seq(0, 1, length = g + 1))
  coupe[1] <- coupe[1] - 0.001
  grp  <- cut(yhat, breaks = coupe, labels = FALSE)
  Obs1 <- tapply(y,    grp, sum)
  Exp1 <- tapply(yhat, grp, sum)
  n_g  <- tapply(y,    grp, length)
  Obs0 <- n_g - Obs1
  Exp0 <- n_g - Exp1
  X2   <- sum((Obs1 - Exp1)^2 / Exp1) + sum((Obs0 - Exp0)^2 / Exp0)
  p    <- 1 - pchisq(X2, df = g - 2)
  cat("=== TEST DE HOSMER-LEMESHOW ===\n")
  cat("Statistique X2    :", round(X2, 4), "\n")
  cat("Degres de liberte :", g - 2, "\n")
  cat("p-value           :", round(p, 4), "\n")
  if (p > 0.05) cat("=> Bonne adequation (p > 0.05)\n") else
    cat("=> Manque d adequation (p < 0.05)\n")
  invisible(list(X2 = X2, df = g - 2, p = p))
}

pred_train <- predict(model_final, type = "response")
hoslem_test(train$DEATH_EVENT_num, pred_train)


# ---------------------------------------------------------------
# ETAPE 7 : ANALYSE DES RESIDUS
# ---------------------------------------------------------------
# --- Graphiques diagnostiques standard ---
par(mfrow = c(2, 2))
plot(model_final,
     pch        = 20,
     col        = adjustcolor("#4E79A7", 0.6),
     sub.caption = "Diagnostics - Modele logistique final")
par(mfrow = c(1, 1))

# --- Residus de deviance ---
res_dev <- residuals(model_final, type = "deviance")
print(round(summary(res_dev), 4))

# Figure sans annotation
plot(fitted(model_final), res_dev,
     pch  = 20,
     col  = adjustcolor("#4E79A7", 0.5),
     xlab = "Probabilites ajustees",
     ylab = "Residus de deviance",
     main = "Residus de deviance vs probabilites ajustees (TRAIN)")
abline(h   = c(-2, 0, 2),
       col = c("red", "gray40", "red"),
       lty = c(2, 1, 2))

# Figure avec identification des points aberrants
idx_aberrants <- which(abs(res_dev) > 2)
text(fitted(model_final)[idx_aberrants],
     res_dev[idx_aberrants],
     labels = idx_aberrants,
     pos = 3, cex = 0.7, col = "red")
cat("Observations avec |residus| > 2 :", idx_aberrants, "\n")

# --- Residus de Pearson ---
res_pear <- residuals(model_final, type = "pearson")
print(round(summary(res_pear), 4))

# --- Distance de Cook ---
cook       <- cooks.distance(model_final)
seuil_cook <- 4 / nrow(train)
pts_cook   <- which(cook > seuil_cook)

cat("Points influents (Cook > 4/n =", round(seuil_cook, 4), ") :", length(pts_cook), "\n")
cat("Indices :", pts_cook, "\n")

plot(cook,
     type = "h",
     col  = ifelse(cook > seuil_cook, "#E15759", "#4E79A7"),
     main = "Distance de Cook - Echantillon TRAIN",
     ylab = "Distance de Cook",
     xlab = "Observation")
abline(h = seuil_cook, col = "red", lty = 2)

# --- Leviers ---
lev       <- hatvalues(model_final)
p_coef    <- length(coef(model_final))
seuil_lev <- 2 * p_coef / nrow(train)
pts_lev   <- which(lev > seuil_lev)
cat("Points a levier eleve (seuil = 2p/n =", round(seuil_lev, 4), ") :", length(pts_lev), "\n")

# --- VIF ---
print(round(vif(model_final), 3))
cat("=> VIF < 5 : pas de multicollinearite\n")

# ---------------------------------------------------------------
# ETAPE 8 : MATRICES DE CONFUSION
# ---------------------------------------------------------------

prob_test         <- predict(model_final,   newdata = test, type = "response")
prob_test_complet <- predict(model_complet, newdata = test, type = "response")

afficher_mat_conf <- function(y_obs, prob, seuil = 0.5, nom = "") {
  pred_cl      <- ifelse(prob >= seuil, 1, 0)
  mat          <- table(Observe = y_obs, Predit = pred_cl)
  mat_complete <- matrix(0, nrow = 2, ncol = 2,
                         dimnames = list(c("0","1"), c("0","1")))
  for (i in rownames(mat))
    for (j in colnames(mat))
      mat_complete[i, j] <- mat[i, j]

  VN  <- mat_complete["0","0"]
  FP  <- mat_complete["0","1"]
  FN  <- mat_complete["1","0"]
  VP  <- mat_complete["1","1"]
  acc <- round((VP + VN) / sum(mat_complete) * 100, 2)
  sens <- round(ifelse((VP + FN) > 0, VP / (VP + FN), NA) * 100, 2)
  spec <- round(ifelse((VN + FP) > 0, VN / (VN + FP), NA) * 100, 2)
  vpp  <- round(ifelse((VP + FP) > 0, VP / (VP + FP), NA) * 100, 2)
  vpn  <- round(ifelse((VN + FN) > 0, VN / (VN + FN), NA) * 100, 2)

  cat("\n--- Matrice de confusion :", nom, "(seuil =", seuil, ")---\n")
  cat(sprintf("                Predit 0   Predit 1\n"))
  cat(sprintf("  Observe 0  %9d  %9d\n", VN, FP))
  cat(sprintf("  Observe 1  %9d  %9d\n", FN, VP))
  cat(sprintf("  Accuracy          : %6.2f %%\n", acc))
  cat(sprintf("  Sensibilite       : %6.2f %%\n", sens))
  cat(sprintf("  Specificite       : %6.2f %%\n", spec))
  cat(sprintf("  Val. pred. pos.   : %6.2f %%\n", vpp))
  cat(sprintf("  Val. pred. neg.   : %6.2f %%\n", vpn))

  invisible(list(mat = mat_complete, acc = acc, sens = sens,
                 spec = spec, vpp = vpp, vpn = vpn))
}

r_train  <- afficher_mat_conf(train$DEATH_EVENT_num, pred_train,
                               seuil = 0.5, nom = "TRAIN - Modele final")
r_test_f <- afficher_mat_conf(test$DEATH_EVENT_num,  prob_test,
                               seuil = 0.5, nom = "TEST  - Modele final")
r_test_c <- afficher_mat_conf(test$DEATH_EVENT_num,  prob_test_complet,
                               seuil = 0.5, nom = "TEST  - Modele complet")

# ---------------------------------------------------------------
# ETAPE 9 : COURBES ROC ET COMPARAISON
# ---------------------------------------------------------------

roc_train  <- roc(train$DEATH_EVENT_num, pred_train,         quiet = TRUE)
roc_test_f <- roc(test$DEATH_EVENT_num,  prob_test,          quiet = TRUE)
roc_test_c <- roc(test$DEATH_EVENT_num,  prob_test_complet,  quiet = TRUE)

auc_train  <- round(auc(roc_train),  4)
auc_test_f <- round(auc(roc_test_f), 4)
auc_test_c <- round(auc(roc_test_c), 4)

cat(sprintf("%-40s : %.4f\n", "Train  - Modele final",   auc_train))
cat(sprintf("%-40s : %.4f\n", "Test   - Modele final",   auc_test_f))
cat(sprintf("%-40s : %.4f\n", "Test   - Modele complet", auc_test_c))

plot(roc_train,
     col         = "#4E79A7",
     lwd         = 2.5,
     legacy.axes = TRUE,
     main        = "Courbes ROC - Comparaison des modeles")
lines(roc_test_f, col = "#E15759", lwd = 2.5)
lines(roc_test_c, col = "#59A14F", lwd = 2,  lty = 2)
abline(a = 0, b = 1, col = "gray60", lty = 3, lwd = 1.5)
legend("bottomright",
       legend = c(
         paste("Train  | Modele final   | AUC =", auc_train),
         paste("Test   | Modele final   | AUC =", auc_test_f),
         paste("Test   | Modele complet | AUC =", auc_test_c)
       ),
       col = c("#4E79A7", "#E15759", "#59A14F"),
       lwd = c(2.5, 2.5, 2),
       lty = c(1, 1, 2),
       bty = "n",
       cex = 0.9)

# --- Test de DeLong ---
print(roc.test(roc_test_f, roc_test_c))
