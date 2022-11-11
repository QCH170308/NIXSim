function [I] = Double_SPICE(Vin,W,cbsize,rw,nbits)

NOR=cbsize;
Rw=rw;
Vdd=1;
NOC=NOR;
T=1e-9
Vb=0
RL=Rw

Gmin=1e-6; 
Gmax=1e-3; 

W=double(W');
Vin=double(Vin');

tic
[Wp,Wn]=QuantizedWeightsAssign(W,nbits,Gmax,Gmin);

Ip=VMM_SPICE(Vin,Wp,Rw,RL,NOR,NOC,Vb,T);
In=VMM_SPICE(Vin,Wn,Rw,RL,NOR,NOC,Vb,T);
I=(Ip-In)./Gmax;
toc

t=toc
fid=fopen(['~/Desktop/sim_results/SPICE_linear_double/',num2str(cbsize),'-',num2str(rw),'-',num2str(nbits),'b.txt'],'a');
fprintf(fid,'%s\n',num2str(t));
fclose(fid);