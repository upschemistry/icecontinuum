
% Ntot  NQLL   Unc_NQLL
%
% sin equation:
%
%  Ntotlist_theory = np.linspace(0,1)
%
%  Nbar = 1.00
%  Nstar = 0.1 
%  Nstar = .9/(2*np.pi)
%  NQLLlist_theory = -np.sin(2*np.pi*(Ntotlist_theory))*Nstar + Nbar
%
%
%
%0	1.0375	0
data = [...
    0.8	1.108333333	0.070833333
0.2	0.933333333	0.045833333
0.5	1.079166667	0.054166667
0.5	1.05	0.054166667
0.870833333	1	0.05
0.129166667	1.129166667	0.058333333
0.279166667	0.916666667	0.041666667
0.716666667	1.158333333	0.066666667
0.216666667	0.879166667	0.041666667
0.783333333	1.1375	0.0625
0.0125	0.9375	0.045833333
0.9875	1.0875	0.066666667
0.004166667	1.083333333	0.058333333
0.995833333	0.954166667	0.05
0.7875	1.108333333	0.070833333
0.2125	0.933333333	0.045833333
0.616666667	1.0875	0.054166667
0.379166667	1	0.05
0.058333333	1.05	0.05
0.941666667	1.066666667	0.054166667
0.004166667	1.033333333	0.054166667
0.995833333	1.083333333	0.054166667
0.758333333	1.120833333	0.066666667
0.245833333	0.958333333	0.045833333
0.5	1.066666667	0.0625
0.5	1.066666667	0.058333333
0.554166667	1.075	0.054166667
0.45	1.0375	0.054166667];

Ntot = data(:,1);
Nqll = data(:,2);
errors = data(:,3);

 Ntotlist_theory = linspace(0,1);

Nbar = 1.03 ;
%Nstar = .9/(2*pi) ;
Nstar = 0.12 ;
NQLLlist_theory = -sin(2*pi*(Ntotlist_theory))*Nstar + Nbar ;


%Nqll_rand = randn(size(Ntot));
%[hrand,prand,strand] = chi2gof(Nqll_rand)

Nqll_theory = -sin(2*pi*(Ntot))*Nstar + Nbar ;
[h,p,st] = chi2gof(Nqll-Nqll_theory)
plot(Ntot,Nqll,'o',Ntotlist_theory,NQLLlist_theory)
hold on
errorbar(Ntot,Nqll,errors,'k.')
hold off
%[P,S,Mu]=polyfit(Ntot,Nqll,25);
%plot(Ntot,Nqll,'o',Ntot,polyval(P,Ntot),'.')
