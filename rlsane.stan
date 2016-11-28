data {
	int<lower=1> T;		          //number of trials
	int<lower=1> S;		          //number of sessions
	int<lower=1> L[S];	        //trials per session
	int<lower=0,upper=2> C[T];	//choice in each trial -- 1 straight, 2 swerve, 0 no choice
	real Vsfe;		              //payoff in front of wall
	real Vstr[T];               //straight payoff
	real Vcop[T];               //cooperative swerve payoff
	real R[T];		              //recieved reward
	int<lower=0,upper=1> QI;    //gate RL influence on utility
	int<lower=0,upper=1> KI;    //gate choice autocorrelation
	int<lower=0,upper=1> VI;    //gate value
}

transformed data{
  int Cb[T];
  for (t in 1:T) Cb[t] = C[t]-1;
}

parameters {
  real<lower=0,upper=1> p;      //probability of opponent swerving
	real<lower=0,upper=1> alpha;	//reward learning rate
	real<lower=0,upper=1> tau;	  //choice effect decay 
	real Q0;                      //initial value
	real beta_q;		              //reward weight
	real beta_k;		              //choice history weight
	real beta_v;		              //payoff weight
	real b;				                //bias
}

transformed parameters {
	real Q[T,2];
	real K[T,2];
	real U[T];
	
	{
		int nxtsess;
		int sess;
		nxtsess = 1;
		sess = 0;

		for (t in 1:T) {
			real w;
			
			if (t == nxtsess) {
				for (i in 1:2) {
					Q[t,i] = Q0;
					K[t,i] = 0;
				}
				sess = sess + 1;
				nxtsess = nxtsess + L[sess];
			} else if (C[t-1]==0) {
				for (i in 1:2) {
					Q[t,i] = Q[t-1,i];
					K[t,i] = K[t-1,i]*(1-tau);
				}
			} else {
				for (i in 1:2) {
					Q[t,i] = Q[t-1,i] + ((C[t-1]==i) ? 
						alpha*(R[t-1] - Q[t-1,i]) : 0);
					K[t,i] = K[t-1,i]*(1-tau) + 
						((C[t-1]==i) ? 1 : 0);
				}
			}

			U[t] = beta_v*( (1-p)*Vsfe + p*Vcop[t] - p*Vstr[t])*VI + 
				beta_q*(Q[t,2]-Q[t,1])*QI + beta_k*(K[t,2]-K[t,1])*KI + b;
		}
	}
}

model {
	Cb ~ bernoulli_logit(U);
	p ~ beta(2,2);
	alpha ~ beta(2,2);
	tau ~ beta(2,2);
	beta_q ~ normal(2,10);
	beta_k ~ normal(2,10);
	beta_v ~ normal(2,10);
	b ~ normal(0,10);
	Q0 ~ normal(1,5);
}
