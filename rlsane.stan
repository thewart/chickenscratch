data {
	int<lower=1> T;		//number of trials
	int<lower=1> S;		//number of sessions
	int<lower=1> L[s];	//trials per session
	int<lower=-1> C[t];	//choice in each trial
	int V[t,2];		//payoffs per trial
}

parameters {
	real<lower=0,upper=1> alpha;	//reward learning rate
	real<lower=0,upper=1> tau;	//choice effect decay 
	real<lower=0> beta_q;		//reward weight
	real<lower=0> beta_k;		//choice history weight
	real<lower=0> beta_v;		//payoff weight
	real b;				//bias
}

transformed parameters {
	real Q[t,2];
	real K[t,2];
	real U[t];
	
	{
		int nxtsess;
		int sess;
		nxtsess <- 1;
		sess <- 0;

		for (t in 1:T) {
			
			if ( t == nxtsess) {
				for (i in 1:2) {
					Q[t,i] <- 0;
					K[t,i] <- 0;
				}
				sess <- sess + 1;
				nxtsess <- nxtsess + L[sess];
			} else if (C[t-1]==-1) {
				for (i in 1:2) {
					Q[t,i] <- Q[t-1,i];
					K[t,i] <- K[t-1,i]*(1-tau);
				}
			} else {
				for (i in 1:2) {
					Q[t,i] <- Q[t-1,i] + (C[t-1]==(i-1)) ? 
						alpha*(V[t-1,C[t-1]+1] - Q[t-1,i]) : 0;
					K[t,i] <- K[t-1,i]*(1-tau) + 
						(C[t-1]==(i-1)) ? 1 : 0;
				}
			}
			
			U[t] = beta_v*(V[t,1]-V[t,2]) beta_q*(Q[t,1]-Q[t,2]) +
				beta_k*(K[t,1]-K[t,2]) + b;
		}
	}
}

model {
	C ~ bernoulli_logit(U);
	alpha ~ beta(2,2);
	tau ~ beta(2,2);
	beta_r ~ normal(2,10);
	beta_c ~ normal(2,10);
	beta_v ~ normal(2,10);
}
