data {
  int<lower=1> T;             //number of trials
  int<lower=1> S;             //number of sessions
  int<lower=1> L[S];          //trials per session
  int<lower=0,upper=2> C1[T]; //choice in each trial -- 1 straight, 2 swerve, 0 no choice
  int<lower=0,upper=2> C2[T]; //opponent's choice
  real Vsfe;                  //payoff in front of wall
  real Vstr[T];               //straight payoff
  real Vcop[T];               //cooperative swerve payoff
  real R1[T];                 //recieved reward
  int<lower=0,upper=1> H[T];  //0 low coherence 1 high coherence
  int<lower=0,upper=1> QI;    //gate RL influence on utility
  int<lower=0,upper=1> KI;    //gate choice autocorrelation
  int<lower=0,upper=1> VI;    //gate value
}

parameters {
  real<lower=0,upper=1> alpha;    //reward learning rate
  real<lower=0,upper=1> tau;      //choice effect decay 
  real<lower=0> Qinit;            //initial value
  real beta_0[2];                 //bias towards swerve
  real beta_q[2];                 //RL value weight
  real beta_k[2];                 //choice history weight
  real beta_v[2];                 //Current trial payoff weight
  
  real init_0_opp;
  real init_coh_opp;
  real init_v_opp;
  real init_vbycoh_opp;
  real<lower=0> eta[2,2];
  // real<lower=0> eta_0_opp;
  // real<lower=0> eta_coh_opp;
  // real<lower=0> eta_v_opp;
  // real<lower=0> eta_vbycoh_opp;
}

transformed parameters {
  real U[T];                    //utility!
  real Q[T,2];                  //RL estimated value
  real K[T,2];                  //perseveration component
  real beta_0_opp[T];             //opponent's swerve bias
  real beta_coh_opp[T];
  real beta_v_opp[T];             //opponent's value influence
  real beta_vbycoh_opp[T];

  {
    real Popp;
    real Vdiff;
    int nxtsess;
    int sess;
    real pe;
    nxtsess = 1;
    sess = 0;

    for (t in 1:T) {

      //update learned values based on what happened on t-1
      if (t == nxtsess) { //reset to initial values at new session
        sess = sess + 1;
        nxtsess = nxtsess + L[sess];
        for (i in 1:2) {
          Q[t,i] = Qinit;
          K[t,i] = 0;
        }
        beta_0_opp[t] = init_0_opp;
        beta_coh_opp[t] = init_coh_opp;
        beta_v_opp[t] = init_v_opp;
        beta_vbycoh_opp[t] = init_vbycoh_opp;

      } else if (C1[t-1]==0) { //t-1 was a P2 only catch trial
        for (i in 1:2) {
          Q[t,i] = Q[t-1,i];
          K[t,i] = K[t-1,i];
        }
        beta_0_opp[t] = beta_0_opp[t-1];
        beta_coh_opp[t] = beta_coh_opp[t-1];
        beta_v_opp[t] = beta_v_opp[t-1];
        beta_vbycoh_opp[t] = beta_vbycoh_opp[t-1];
        
      
      } else { //if t-1 was a 2-player trial or a P1 only catch trial
        
        for (i in 1:2) { //update Q and K no matter what
          Q[t,i] = Q[t-1,i] + ((C1[t-1]==i) ? 
            alpha*(R1[t-1] - Q[t-1,i]) : 0);
          K[t,i] = K[t-1,i]*(1-tau) + 
            ((C1[t-1]==i) ? 1 : 0);
        }
        
        if (C2[t-1]!=0) { //if not P1 catch trial, update beliefs about opponent
          pe = C2[t-1]-1 - Popp; //this is Popp[t-1]
          if (H[t-1]==0) {
            beta_0_opp[t] = beta_0_opp[t-1] + eta[1,1] * pe;
            beta_coh_opp[t] = beta_coh_opp[t-1] + eta[1,2] * pe*0;
            beta_v_opp[t] = beta_v_opp[t-1] + eta[1,1] * (Vcop[t-1]-Vstr[t-1]) * pe;
            beta_vbycoh_opp[t] = beta_vbycoh_opp[t-1] + eta[1,2] * (Vcop[t-1]-Vstr[t-1]) * pe;
          } else {
            beta_0_opp[t] = beta_0_opp[t-1] + eta[2,1] * pe;
            beta_coh_opp[t] = beta_coh_opp[t-1] + eta[2,2] * pe;
            beta_v_opp[t] = beta_v_opp[t-1] + eta[2,1] * (Vcop[t-1]-Vstr[t-1]) * pe;
            beta_vbycoh_opp[t] = beta_vbycoh_opp[t-1] + eta[2,2] * (Vcop[t-1]-Vstr[t-1]) * pe;
          }
        } else { //otherwise no update
          beta_0_opp[t] = beta_0_opp[t-1];
          beta_coh_opp[t] = beta_coh_opp[t-1];
          beta_v_opp[t] = beta_v_opp[t-1];
          beta_vbycoh_opp[t] = beta_vbycoh_opp[t-1];
        }
      }
      
      //combined utility
      if (C2[t]!=0) {
        Popp = inv_logit(beta_0_opp[t] + beta_coh_opp[t]*(H[t]-0.5) + 
          beta_v_opp[t]*(Vcop[t]-Vstr[t]) + beta_vbycoh_opp[t]*(H[t]-0.5)*(Vcop[t]-Vstr[t]));
        Vdiff = (1-Popp)*Vsfe + Popp*(Vcop[t]-Vstr[t]);
      } else {
        Vdiff = Vsfe - Vstr[t];
      }
        
      U[t] = beta_v[H[t]+1]*Vdiff*VI + 
        beta_q[H[t]+1]*(Q[t,2]-Q[t,1])*QI + 
        beta_k[H[t]+1]*(K[t,2]-K[t,1])*KI + 
        beta_0[H[t]+1];
      
    }
  }
}

model {
  for (t in 1:T)
    if (C1[t]>0) (C1[t]-1) ~ bernoulli_logit(U[t]);

  // P0 ~ beta(2,2);
  // alpha ~ beta(2,2);
  // tau ~ beta(2,2);
  // beta_q ~ normal(2,10);
  // beta_k ~ normal(2,10);
  // beta_v ~ normal(2,10);
  // beta_v_opp ~ normal(2,10);
  // b ~ normal(0,10);
  // Q0 ~ normal(1,5);
}

// generated quantities {
//   real loglik;
//   
//   loglik = 0;
//   for (t in 1:T) {
//     if (C1[t]>0) loglik = loglik + bernoulli_logit_lpmf(C1[t]-1 | U[t]);
//   }
// }
