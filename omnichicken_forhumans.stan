data {
  int<lower=1> T;             //number of trials
  int<lower=0,upper=2> C1[T]; //choice in each trial -- 1 straight, 2 swerve, 0 no choice
  int<lower=0,upper=2> C2[T]; //opponent's choice
  real Vsfe;                  //payoff in front of wall
  real Vstr[T];               //straight payoff
  real Vcop[T];               //cooperative swerve payoff
  real Vstr_opp[T];           //straight payoff for opponent
  real Vcop_opp[T];           //cooperative payoff for opponent
  real R1[T];                 //recieved reward
  int<lower=0,upper=1> H[T];  //0 low coherence 1 high coherence
  int<lower=0,upper=1> QI;    //gate RL influence on utility -- beta_q, alpha, Q0
  int<lower=0,upper=1> CI;    //gate choice autocorrelation -- beta_k, kappa
  int<lower=0,upper=1> VI;    //gate value -- beta_v
  int<lower=0,upper=1> GI;    //gate generalization across hi and low coherence trials -- eta off-diagonal
  int<lower=0,upper=1> LI;    //gate learning -- eta
  int<lower=0,upper=1> KI;    //gate "ToM" -- init_v_opp, init_vbycoh_opp, and the associated betas
  int<lower=0,upper=1> OI;    //gate outcome preference -- sum of coop and crash bonuses
  int<lower=0,upper=1> SI;    //gate social utility
}

parameters {
  real<lower=0,upper=1> alpha;    //reward learning rate
  real<lower=0,upper=1> tau;      //choice effect decay 
  real beta_0;                 //bias towards swerve
  real<lower=0> beta_q;        //RL value weight
  real beta_k;                 //choice history weight
  real<lower=0> beta_v;        //Current trial payoff weight
  real beta_soc;             //Social utility for rewards from swerving
  
  real init_0_opp;
  real init_coh_opp;
  real init_v_opp;
  real init_vbycoh_opp;
  real<lower=0> eta[2];
  
  real Vo;                         //coop bonus+crash penalty
}

model {
  real U;                    //utility!
  real Q[2];                  //RL estimated value
  real K[2];                  //perseveration component
  real beta_0_opp;             //opponent's swerve bias
  real beta_coh_opp;
  real beta_v_opp;             //opponent's value influence
  real beta_vbycoh_opp;
  real Popp;
  real Vdiff;
  real Vdiff_soc;
  real pe;
  real rpe;
  real kpe;

  for (t in 1:T) {

    //update learned values based on what happened on t-1
    if (t == 1) { //reset to initial values at new session
      
      for (i in 1:2) {
        Q[i] = 0;
        K[i] = 0;
      }
      beta_0_opp = init_0_opp;
      beta_coh_opp = init_coh_opp;
      beta_v_opp = init_v_opp * KI;
      beta_vbycoh_opp = init_vbycoh_opp * KI;
      
    } else if (C1[t-1]!=0) { //if t-1 was a 2-player trial or a P1 only catch trial
      
      rpe = R1[t-1] - Q[C1[t-1]]; //update Q and K no matter what
      kpe = 1-K[C1[t-1]];
      for (i in 1:2) { 
        Q[i] += (C1[t-1]==i) ? alpha*rpe : 0;
        K[i] += (C1[t-1]==i) ? tau*kpe : 0;
      }
        
      if (C2[t-1]!=0) { //if not P1 catch trial, update beliefs about opponent
        pe = C2[t-1]-1 - Popp;
        if (H[t-1]==0) {
          beta_0_opp += eta[1] * pe * LI;
          beta_coh_opp += eta[2] * pe * GI * LI;
          beta_v_opp += eta[1] * (Vcop_opp[t-1]-Vstr_opp[t-1]) * pe * LI * KI;
          beta_vbycoh_opp += eta[2] * (Vcop_opp[t-1]-Vstr_opp[t-1]) * pe * GI * LI * KI;
        } else {
          beta_0_opp += eta[2] * pe * GI * LI;
          beta_coh_opp += eta[1] * pe * LI;
          beta_v_opp += eta[2] * (Vcop_opp[t-1]-Vstr_opp[t-1]) * pe * GI * LI * KI;
          beta_vbycoh_opp += eta[1] * (Vcop_opp[t-1]-Vstr_opp[t-1]) * pe * LI * KI;
        }
      } //else if (C2[t-1]==0) { } //otherwise no update
    } //else if (C1[t-1]==0) { } //t-1 was a P2 only catch trial, no updates
      
    //combined utility
    if (C2[t]!=0) {
      Popp = inv_logit(beta_0_opp*(H[t]==0) + beta_coh_opp*(H[t]==1) + 
        beta_v_opp*(H[t]==0)*(Vcop_opp[t]-Vstr_opp[t]) + 
        beta_vbycoh_opp*(H[t]==1)*(Vcop_opp[t]-Vstr_opp[t]));
      Vdiff = (1-Popp)*Vsfe + Popp*(Vcop[t]-Vstr[t]) + Popp*Vo*OI;
      Vdiff_soc = Popp*Vcop_opp[t] + (1-Popp)*Vstr_opp[t] - Popp*Vsfe;
    } else {
      Vdiff = Vsfe - Vstr[t];
      Vdiff_soc = 0;
    }
      
    if (C1[t]>0) {  
      U = beta_v*Vdiff*VI + 
        beta_soc*Vdiff_soc*SI +
        beta_q*(Q[2]-Q[1])*QI + 
        beta_k*(K[2]-K[1])*CI + 
        beta_0;
      (C1[t]-1) ~ bernoulli_logit(U);
    }
  }
}
