function [Wp,Wn] = QuantizedWeightsAssign(W,nbits,Gmax,Gmin)
nLevels=2^(nbits);
Wp=Gmin*ones(size(W));
Wn=Gmin*ones(size(W));
[m,n]=size(W);
WQ=zeros(m,2*n);

for k=1:nLevels/2
    Mp=double(W==2*double(k)/double(nLevels));
    Mn=double(W==-2*double(k)/double(nLevels));
    Wp=Wp+Mp*2*double(k)/double(nLevels)*(Gmax-Gmin);
    Wn=Wn+Mn*2*double(k)/double(nLevels)*(Gmax-Gmin);
end