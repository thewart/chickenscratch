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



##### compare to Rani's fits
tbl <- fread("~/Downloads/Rani_tbl.csv")
tbli <- tbl[1,-1] %>% unlist()

eval_loglik <- function(model,parvec) {
  cpar <- list(alpha=parvec[1],tau=parvec[2],Qinit=parvec[3],beta_0=parvec[4:5],
               beta_q=parvec[6:7],beta_k=parvec[8:9],beta_v=parvec[10:11],
               init_0_opp=parvec[12],init_coh_opp=parvec[13],init_v_opp=parvec[14],
               init_vbycoh_opp=parvec[15],eta=matrix(parvec[15:18],2),Vo=parvec[20])
  stanobj <- sampling(model,standat,chains=0)
  
  return(log_prob(stanobj,unconstrain_pars(stanobj,cpar)))
}
