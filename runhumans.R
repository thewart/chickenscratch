library(rstan) #load rstan into namespace
library(data.table) #data table is magical!
rstan_options(auto_write = TRUE) #make rstan write compiled models to disk for later use
chkn <- stan_model("code/chickenscratch/omnichicken_forhumans.stan") #compile or load model
bigdat <- fread("~/Downloads/wbl1184.csv") #read in data as data.table

nstart <- 20 #number of random initializations for each subject
bigdat[,R1:=R1/30] #normalize R1 to match V units, then replace R1 w/ normed values
bigdat[,R2:=R2/30] #same for R2
submodel <- list(QI=1,VI=1,CI=1, #specify submodel
                GI=0,KI=0,LI=0,OI=0,SI=0,
                Vsfe=1)

idat <- bigdat[Ses==2,!"Ses",with=F] #pull out data from Ses 1, without session number
standat <- c(list(T=nrow(idat)), #combine lists into the data to pass to stan, and also add trial number
                  as.list(idat),
                  submodel)

#fit 20 times and return best model
bestll <- -Inf
for (j in 1:nstart) {
  candidate <- optimizing(chkn,standat,verbose=2)
  if (candidate$value > bestll) {
    fit <- candidate
    bestll <- fit$value
  }
}
