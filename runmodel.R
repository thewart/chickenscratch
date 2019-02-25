modelopts <- function(mymod=list()) 
  return(modifyList(list(QI=1,CI=1,KI=1,VI=1,GI=1,LI=1,OI=1),mymod))

read_chickendat <- function(path,mymod=modelopts()) {
  standat <- as.list(fread(path))
  standat$L <- table(standat$Sn) %>% as.vector()
  standat$Vsfe <- 0.3
  standat$T <- length(standat$C1)
  standat$S <- length(standat$L)
  return(standat <- c(standat,mymod))
}

# chmd <- stan_model("~/code/chickenscratch/omnichicken.stan")
# fit <- optimizing(chmd,standat,verbose=T,as_vector=F,refresh=100)

##### compare to Rani's fits
# tbl <- fread("~/Downloads/Rani_tbl.csv")
# tbli <- tbl[1,-1] %>% unlist()

eval_loglik_uncvector <- function(model,standat,parvec) {
  cpar <- list(alpha=parvec[1],tau=parvec[2],Qinit=parvec[3],beta_0=parvec[4:5],
               beta_q=parvec[6:7],beta_k=parvec[8:9],beta_v=parvec[10:11],
               init_0_opp=parvec[12],init_coh_opp=parvec[13],init_v_opp=parvec[14],
               init_vbycoh_opp=parvec[15],eta=matrix(parvec[16:19],2),Vo=parvec[20])
  stanobj <- sampling(model,standat,chains=0)
  
  return(log_prob(stanobj,unconstrain_pars(stanobj,cpar),adjust_transform=F))
}

