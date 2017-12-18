data {
  int<lower=1> T;             //number of trials
  int<lower=1> S;             //number of sessions
  int<lower=1> L[S];          //trials per session
  int<lower=0,upper=2> C1[T];  //choice in each trial -- 1 straight, 2 swerve, 0 no choice
  int<lower=0,upper=2> C2[T];  //opponent's choice
  real Vsfe;                  //payoff in front of wall
  real Vstr[T];               //straight payoff
  real Vcop[T];               //cooperative swerve payoff
  real R1[T];                  //recieved reward
  int<lower=0,upper=1> H[T];  //0 low coherence 1 high coherence
  int<lower=0,upper=1> QI;    //gate RL influence on utility
  int<lower=0,upper=1> KI;    //gate choice autocorrelation
  int<lower=0,upper=1> VI;    //gate value
  int<lower=0,upper=1> PI;    //gate swerve prob learning
}

parameters {
  real<lower=0,upper=1> lambda;   //swerve learning rate
  real<lower=0,upper=1> alpha;    //reward learning rate
  real<lower=0,upper=1> tau;      //choice effect decay 
  real<lower=0> Q0;                //initial value
  real<lower=0,upper=1> P0[2];    //initial prob of opponent swerving
  real beta_q[2];                    //RL value weight
  real beta_k[2];                    //choice history weight
  real beta_v[2];                    //Current trial payoff weight
  real beta_0[2];                         //bias towards swerve
}

transformed parameters {


  real U[T];                    //utility!
  
  {
    real Q[T,2];                  //RL estimated value
    real K[T,2];                  //perseveration component
    real Popp[T,2];                  //swerve prob for incoherent (1) and coherent (2) trials 
    int nxtsess;
    int sess;
    nxtsess = 1;
    sess = 0;

    for (t in 1:T) {
      real w;
      
	  if (t == nxtsess) { //reset to initial values at new session
        sess = sess + 1;
        nxtsess = nxtsess + L[sess];
        for (i in 1:2) {
          Q[t,i] = Q0;
          K[t,i] = 0;
          Popp[t,i] = P0[i];
        }

      } else if ( (C1[t-1]==0) || (C2[t-1]==0) ) { //t-1 was control trial, nothing learned
        for (i in 1:2) {
          Q[t,i] = Q[t-1,i];
          K[t,i] = K[t-1,i]*(1-tau);
          Popp[t,i] = Popp[t-1,i];
        }
        
      } else {
        for (i in 1:2) { //update for successful trials
          Q[t,i] = Q[t-1,i] + ((C1[t-1]==i) ? 
            alpha*(R1[t-1] - Q[t-1,i]) : 0);
          K[t,i] = K[t-1,i]*(1-tau) + 
            ((C1[t-1]==i) ? 1 : 0);
          Popp[t,i] = Popp[t-1,i] + ((H[t-1]==i) ?
          lambda*( (C2[t-1]-1) - Popp[t-1,i])*PI : 0);

        }

      }

      //combined utility
      U[t] = beta_v[H[t]+1]*( (1-Popp[t,H[t]+1])*Vsfe + Popp[t,H[t]+1]*(Vcop[t]-Vstr[t]) )*VI + 
        beta_q[H[t]+1]*(Q[t,2]-Q[t,1])*QI + beta_k[H[t]+1]*(K[t,2]-K[t,1])*KI + beta_0[H[t]+1];
    }
  }
}

model {
  for (t in 1:T)
    if ( (C1[t]>0) && (C2[t]>0) ) (C1[t]-1) ~ bernoulli_logit(U[t]);
  // P0 ~ beta(2,2);
  // lambda ~ beta(2,2);
  // alpha ~ beta(2,2);
  // tau ~ beta(2,2);
  // beta_q ~ normal(2,10);
  // beta_k ~ normal(2,10);
  // beta_v ~ normal(2,10);
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
