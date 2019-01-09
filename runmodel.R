library(rstan)
rstan_options(auto_write = TRUE)
standat <- as.list(fread("~/Dropbox/chickenscratch_behavior/CH_L_170727.csv"))
standat$L=c(339, 663, 283, 441, 601, 214, 989, 999, 828, 783, 441, 360, 868, 1094, 595, 417, 343, 415, 497, 527, 999, 124, 238, 973, 1176, 491, 289)
standat$Vsfe <- 0.3
standat$T <- length(standat$C1)
standat$S <- length(standat$L)
standat <- c(standat,list(QI=1,CI=1,KI=1,VI=1,GI=1,LI=1,OI=1))

chmd <- stan_model("~/code/chickenscratch/omnichicken.stan")
fit <- optimizing(chmd,standat,verbose=T,as_vector=F,refresh=100)

stanobj <- sampling(chmd,standat,chains=0)
