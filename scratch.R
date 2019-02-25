source("~/code/chickenscratch/runmodel.R")
bpath <- "~/Dropbox/chickenscratch_behavior/"
chmd <- stan_model("~/code/chickenscratch/omnichicken.stan")
setid <- c('BC', 'BH', 'CB','CH', 'CL', 'HB', 'HC', 'LC')

initvals <- function(chain) list(alpha = runif(1),
  tau = runif(1),
  Qinit = abs(rnorm(1,1)),
  beta_0 = rnorm(3)*2+1,
  beta_q = abs(rnorm(3)*2),
  beta_k = rnorm(3)*2+1,
  beta_v = abs(rnorm(3)*2),
  init_0_opp = rnorm(1)*5+5,
  init_coh_opp = rnorm(1)*5+5,
  init_v_opp = rnorm(1)*5+5,
  init_vbycoh_opp = rnorm(1)*5+5,
  eta = abs(rnorm(4)*0.05) %>% matrix(nrow=2)
)

iter <- 100
cores <- 5
stanfitlist <- list()
ll <- matrix(nrow=iter,ncol=length(setid))
for (i in 1:length(setid)) {
  cat(paste0(i,"\n"))
  standat <- read_chickendat(paste0(bpath,setid[i],".csv"))
  cl <- readyparallel(cores)
  system.time(juh <- foreach(1:iter) %dopar% {library(rstan); library(gtools)
    stanfit <- optimizing(chmd,standat,init=initvals())
    return(stanfit)
  })
  ll[,i] <- sapply(juh,function(x) x$val)
  stanfitlist[[i]] <- juh[[which.max(ll[,i])]]
  stopCluster(cl)
}
fitsfull <- stanfitlist
llfull <- ll

iter <- 100
cores <- 5
stanfitlist <- list()
ll <- matrix(nrow=iter,ncol=length(setid))
for (i in 1:length(setid)) {
  cat(paste0(i,"\n"))
  standat <- read_chickendat(paste0(bpath,setid[i],".csv"),modelopts(list(LI=0)))
  cl <- readyparallel(cores)
  system.time(juh <- foreach(1:iter) %dopar% {library(rstan); library(gtools)
    stanfit <- optimizing(chmd,standat,init=initvals())
    return(stanfit)
  })
  ll[,i] <- sapply(juh,function(x) x$val)
  stanfitlist[[i]] <- juh[[which.max(ll[,i])]]
  stopCluster(cl)
}
