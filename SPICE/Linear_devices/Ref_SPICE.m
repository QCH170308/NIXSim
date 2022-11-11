function [I] = Ref_SPICE(Vin,W,cbsize,rw,nbits)

NOR=cbsize;
Rw=rw;
Vdd=1;
NoRefColns=1;
NOC=NOR+NoRefColns;
T=1e-9
Vb=0
RL=Rw

Gmin=1e-6; 
Gmax=1e-3; 

nLevels=2^nbits;
Gref=Gmax/2;
DG=(Gmax-Gmin)/double(nLevels);

W=double(W');
Vin=double(Vin');

[N,M]=size(W);
m=N;
n=M+NoRefColns;

tic
%% % generate array contents
Gp=zeros(m,m);
Gn=zeros(m,m);

for k=1:nLevels/2
    Mp=double(W==2*double(k)/double(nLevels));
    Mn=double(W==-2*double(k)/double(nLevels));
    Gp=Gp+Mp*(Gref+double(k)*DG);
    Gn=Gn+Mn*(Gref-double(k)*DG);
end

G=Gp+Gn;
G(G==0)=Gref;

%Automating ref colns locations
LocRefColns=zeros(1,NoRefColns);
GrfCol=Gref*ones(m,1);

for kk=1:NoRefColns
    LocRefColns(kk)=kk*floor(M/(NoRefColns+1))+kk;  %add [1 2 3 ..] two each location.
end

M=M+NoRefColns;

for ff=1:NoRefColns
     Gm=[G(:,1:LocRefColns(ff)-1) GrfCol G(:,LocRefColns(ff):end)];
end

Is=VMM_SPICE(Vin,Gm,Rw,RL,NOR,NOC,Vb,T);
Io=Is/Gref;

if NoRefColns==1
    %Io_s=TM-TM(:,LocRefColns);
    Io_s=Io-repmat(Io(:,LocRefColns),1,M);
    Io_s(:,LocRefColns)=[];
    I=Io_s;
elseif NoRefColns>1
    if LocRefColns(1)>1
        Io_s(:,1:LocRefColns(1)-1)=Io(:,1:LocRefColns(1)-1)-Io(:,LocRefColns(1));
    end
    for dd=1:NoRefColns-1
        delta=floor((LocRefColns(dd+1)-LocRefColns(dd))/2);
        Io_s(:,LocRefColns(dd)+1:LocRefColns(dd)+delta)=Io(:,LocRefColns(dd)+1:LocRefColns(dd)+delta)...
                     -Io(:,LocRefColns(dd));
        Io_s(:,LocRefColns(dd)+delta+1:LocRefColns(dd+1)-1)=Io(:,LocRefColns(dd)+delta+1:LocRefColns(dd+1)-1)...
                     -Io(:,LocRefColns(dd+1)); 
    end
    Io_s(:,LocRefColns(end)+1:end)=Io(:,LocRefColns(end)+1:end)-Io(:,LocRefColns(end));
 
    for dd=1:NoRefColns
        Io_s(:,LocRefColns(dd)-dd+1)=[]; 
    end
    I=Io_s; 
end
toc

t=toc
fid=fopen(['~/Desktop/sim_results/SPICE_linear_ref/',num2str(cbsize),'-',num2str(rw),'-',num2str(nbits),'b.txt'],'a');
fprintf(fid,'%s\n',num2str(t));
fclose(fid);