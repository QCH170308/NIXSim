function [Io] = VMM_SPICE(Vin,Gm,Rw,RL,NOR,NOC,Vb,T)

%[Vin,~] = PartitionInput(Vin,NOR);
SubcitName=GenerateCrossbarSubCircuitNetlist(Gm,NOR,NOC,Rw);
XbarName=ReadOutputCurrent(Vin,NOR,NOC,RL,Vb,T,SubcitName);
[~,Io_flipped,~]=ngspice([XbarName '.cir']);
delete([XbarName '.cir']);
delete([SubcitName '.sub']);
Io=fliplr(Io_flipped(:,10)');
Io=Io(1:NOC);
end

