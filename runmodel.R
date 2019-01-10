library(rstan)
rstan_options(auto_write = TRUE)
standat <- as.list(fread("~/Dropbox/chickenscratch_behavior/CH_L_170727.csv"))
standat$L=c(339, 663, 283, 441, 601, 214, 989, 999, 828, 783, 441, 360, 868, 1094, 595, 417, 343, 415, 497, 527, 999, 124, 238, 973, 1176, 491, 289)
standat$Vsfe <- 0.3
standat$T <- length(standat$C1)
standat$S <- length(standat$L)
standat <- c(standat,list(QI=1,CI=1,KI=1,VI=1,GI=1,LI=1,OI=1))

chmd <- stan_model("~/code/chickenscratch/omnichicken.stan")

stanobj <- sampling(chmd,standat,chains=0)

log_prob()

tbl <- fread("~/Downloads/Rani_tbl.csv")
tbli <- tbl[1,-1] %>% as.vector()
cpar <- list(alpha=tbli[1],tau=tbli[2],Qinit=tbli[3],beta_0=tbli[4:5],
             beta_q=tbli[6:7],beta_k=tbli[8:9],beta_v=tbli[10:11],
             init_0_opp=tbli[12],init_coh_opp=tbli[13],init_v_opp=tbli[14],init_vbycoh_opp[15],
             eta=matrix(tbli[15:18]),Vo=tbli[20])