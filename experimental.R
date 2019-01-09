mdat <- fread("~/Downloads/CH_L_170727.csv")
mdat[C1>0 & C2>0 & H==0,.(pswerve=mean(C1-1)),by=Vcop-Vstr]

k1ls <- stan_model("~/code/chickenscratch/k1learner_sep.stan")

sess <- rep(1:length(standat$L),standat$L)
sf_k1l <- list()
sf_k1s <- list()
for (i in 1:standat$S) {
  sessdat <- standat
  it <- sess==i
  sessdat$C1 <- sessdat$C1[it]
  sessdat$C2 <- sessdat$C2[it]
  sessdat$Vstr <- sessdat$Vstr[it]
  sessdat$Vcop <- sessdat$Vcop[it]
  sessdat$R1 <- sessdat$R1[it]
  sessdat$R2 <- sessdat$R2[it]
  sessdat$H <- sessdat$H[it]
  sessdat$S <- 1
  sessdat$T <- sessdat$L[i]
  sessdat$L <- array(sessdat$T)
  sf_k1l[[i]] <- optimizing(k1l,sessdat,verbose=T,as_vector=F,refresh=100,init=fit_k1l$par[1:15])
  sf_k1s[[i]] <- optimizing(k1s,sessdat,verbose=T,as_vector=F,refresh=100,init=fit_k1s$par[1:9])
}

standat_sep <- standat
standat_sep$P <- with(standat,Vcop-Vstr+10*H) %>% ordered() %>% as.numeric()
standat_sep$Vdiff <- rep(with(standat,sort(unique(Vcop-Vstr))),2)
standat_sep$H <- rep(c(0,1),each=6)
rla <- stan_model("~/code/chickenscratch/k0learner_sep.stan")
fit_rl <- optimizing(rla,standat_sep,verbose=T,as_vector=F,refresh=100)


### kernel method
bdat <- with(standat,data.table(C1,C2,Vstr,Vcop,H,R1,R2))
bdat$Sess <- rep(1:standat$S,standat$L)
bdat <- bdat[C1>0 & C2>0]
bdat[,TrialType:=ordered(Vcop-Vstr)]
newnames <- bdat[,levels(TrialType) %>% str_replace("-","") %>% paste(rep(c("Straight","Coop"),each=3))]
levels(bdat$TrialType) <- newnames
#bdat[TrialType %in% c("-2.5","-1.5"),TrialType:="-1"]
#bdat[TrialType %in% c("-0.5","0.5"),TrialType:="0"]
#bdat[TrialType %in% c("1.5","2.5"),TrialType:="1"]
#bdat[H==1,TrialType:=paste0(TrialType,"H")]
bdat[C1==1 & C2==1,Outcome:="crash"]
bdat[C1==1 & C2==2,Outcome:="swerve2"]
bdat[C1==2 & C2==1,Outcome:="swerve1"]
bdat[C1==2 & C2==2,Outcome:="coop"]

# ttypes <- bdat[,unique(TrialType)]
# for (k in 1:standat$S) {
#   for (j in 1:length(ttypes)) {
#     tnum <- bdat[,which(TrialType==ttypes[j] & Sess==k)]
#     cname <- paste0("Last",ttypes[j])
#     for (i in 1:(length(tnum)-1)) {
#       lastout <- bdat[,Outcome[tnum[i]]]
#       bdat[(tnum[i]+1):tnum[i+1],(cname):=lastout]
#     }
#     lastlast <- bdat[last(tnum),Outcome]
#     snrow <- last(which(bdat[,Sess==1]))
#     if (last(tnum)<snrow) bdat[(last(tnum)+1):snrow,(cname):=lastlast]
#   }
# }

bdat[2:nrow(bdat),c("LastType","LastOut","LastC1","LastC2","LastR1","LastR2","LastH") := bdat[1:(nrow(bdat)-1),.(TrialType,Outcome,C1-1,C2-1,R1,R2,H)]]
for (i in 1:standat$S) bdat[which(Sess==i) %>% first,c("LastType","LastOut","LastC1","LastC2","LastR1","LastR2","LastH"):=.(NA,NA,NA,NA,NA,NA,NA)]
bdat[LastC1==0,LastR1:=LastR1*-1]
bdat[LastC2==0,LastR2:=LastR2*-1]


glmer(C1-1 ~ H + LastC1 + LastR1 + LastType + LastC2:LastType + (1|Sess),data=bdat[TrialType=="1.5 Straight"],family=binomial) %>% summary()
glmer(C2-1 ~ H + LastC2 + LastType + LastC1:LastType + (1|Sess),data=bdat[TrialType=="-1.5"],family=binomial) %>% summary()

##### P1 C2
pltdat <- bdat[!is.na(LastC2),.(mu=mean(C1-1),ste=sd(C1-1)/sqrt(length(C1))),by=c("TrialType","LastType","LastC2")]
ggplot(pltdat,aes(x=TrialType,y=mu,fill=factor(LastC2))) + geom_bar(stat="identity",position="dodge",width=0.5) + 
  geom_errorbar(aes(ymin=mu-ste,ymax=mu+ste),position=position_dodge(0.5),width=0) + 
  xlab("Current trial payoff condition") + scale_fill_brewer("P2's previous\n action",labels=c("Straight","Swerve"),palette="Set2") +
  facet_wrap(~LastType,ncol=1) + scale_y_continuous("Probability of swerving",minor_breaks = NULL)

pltdat <- bdat[!is.na(LastC2),.(mu=mean(C1-1),ste=sd(C1-1)/sqrt(length(C1))),by=c("TrialType","LastC2")]
ggplot(pltdat,aes(x=TrialType,y=mu,fill=factor(LastC2))) + geom_bar(stat="identity",position="dodge",width=0.5) + 
  geom_errorbar(aes(ymin=mu-ste,ymax=mu+ste),position=position_dodge(0.5),width=0) + 
  ylab("Probability of swerving") + xlab("Current trial payoff condition") + scale_fill_brewer("P2's previous\n action",labels=c("Straight","Swerve"),palette="Set2")


##### P2 C1
pltdat <- bdat[!is.na(LastC1),.(mu=mean(C2-1),ste=sd(C2-1)/sqrt(length(C2))),by=c("TrialType","LastType","LastC1")]
ggplot(pltdat,aes(x=TrialType,y=mu,fill=factor(LastC1))) + geom_bar(stat="identity",position="dodge",width=0.5) + 
  geom_errorbar(aes(ymin=mu-ste,ymax=mu+ste),position=position_dodge(0.5),width=0) + 
  ylab("Probability of swerving") + xlab("Current trial payoff condition") + scale_fill_discrete("P2's previous\n action",labels=c("Straight","Swerve")) +
  facet_wrap(~LastType,ncol=1)

pltdat <- bdat[!is.na(LastC1),.(mu=mean(C2-1),ste=sd(C2-1)/sqrt(length(C2))),by=c("TrialType","LastC1")]
ggplot(pltdat,aes(x=TrialType,y=mu,fill=factor(LastC1))) + geom_bar(stat="identity",position="dodge",width=0.5) + 
  geom_errorbar(aes(ymin=mu-ste,ymax=mu+ste),position=position_dodge(0.5),width=0) + 
  ylab("Probability of swerving") + xlab("Current trial payoff condition") + scale_fill_discrete("P2's previous\n action",labels=c("Straight","Swerve"))


##### P1 RL
bdat[,LastR1cat:=cut(LastR1,breaks=c(-4,-0.1,0.1,0.5,4),labels=c("Straight Reward","Crash","Safe","Coop Reward"),ordered_result = T)]
pltdat <- bdat[!is.na(LastR1cat),.(mu=mean(C1-1),ste=sd(C1-1)/sqrt(length(C1))),by=c("TrialType","LastR1cat")]
ggplot(pltdat,aes(x=TrialType,y=mu,fill=LastR1cat)) + geom_bar(stat="identity",position="dodge",width=0.5) + 
  geom_errorbar(aes(ymin=mu-ste,ymax=mu+ste),position=position_dodge(0.5),width=0) + 
  xlab("Current trial payoff condition") + ylab("Probability of swerving") + scale_fill_discrete("Previous trial reward")

##### P2 RL
bdat[,LastR2cat:=cut(LastR2,breaks=c(-4,-0.1,0.1,0.5,4),labels=c("Straight Reward","Crash","Safe","Coop Reward"),ordered_result = T)]
pltdat <- bdat[!is.na(LastR2cat),.(mu=mean(C2-1),ste=sd(C2-1)/sqrt(length(C2))),by=c("TrialType","LastR2cat")]
ggplot(pltdat,aes(x=TrialType,y=mu,fill=LastR2cat)) + geom_bar(stat="identity",position="dodge",width=0.5) + 
  geom_errorbar(aes(ymin=mu-ste,ymax=mu+ste),position=position_dodge(0.5),width=0) + 
  xlab("Current trial payoff condition") + ylab("Probability of swerving") + scale_fill_discrete("Previous trial reward")
