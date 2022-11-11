function [XbarName]=ReadOutputCurrent(Vin,NOR,NOC,RL,Vb,T,SubcitName)

%%% File generation

[temp1, rand_string] = fileparts(tempname);

XbarName= ['cir',rand_string];

fileName1 = [XbarName '.cir'];
file1 = fopen(fileName1, 'w');

fprintf(file1, ['Xbar#\n\n']);
fprintf(file1, ['*** Read only certain row***\n\n']);
fprintf(file1, ['.include' ' ' SubcitName '.sub\n']);
fprintf(file1, '\n');
fprintf(file1, ['.param RL=' num2str(RL) '\n']);
fprintf(file1, '\n');
fprintf(file1, ['XU' ' ']);

for r=1:1:NOR
    fprintf(file1,[ 'r' num2str(r) 'c'  num2str(0) ' ']);
end
for r=1:1:NOR
    fprintf(file1,[ 'r' num2str(r) 'c'  num2str(NOC) ' ']);
end

for c=1:1:NOC
    fprintf(file1, ['c'  num2str(c) 'r' num2str(0) ' ']);
end
for c=1:1:NOC
    fprintf(file1, ['c'  num2str(c) 'r' num2str(NOR) ' ']);
end

fprintf(file1, [' ' SubcitName ' \n']);

%voltage sources for dirct read
fprintf(file1, ['\n *** voltage sources ***\n']);

for i=1:NOR
    Vi=Vin(i);
    fprintf(file1, [ ['V' 'r' num2str(i) 'c'  num2str(0)] ' ' ['r' num2str(i) 'c'  num2str(0)] ' ' '0'  ' ' num2str(Vi) 'V \n']);
    fprintf(file1, '\n');   
end

%Loads
fprintf(file1, ['\n *** Loads ***\n']);

for j=1:NOC
    fprintf(file1, [['VL'  'r' num2str(0) 'c'  num2str(j)] ' ' ['c'  num2str(j)] ' ' '0' ' ' num2str(Vb) 'V \n']);
    fprintf(file1, [['RL' 'r' num2str(0) 'c'  num2str(j)] ' ' ['c'  num2str(j) 'r' num2str(0) ] ' ' ['c'  num2str(j)] ' RL' '\n']);
end
fprintf(file1,'\n');

%%simulation setups and printing the outputs
fprintf(file1, '\n *** Simulation Setup***\n');
fprintf(file1,['.tran 1e-10' ' ' num2str(T)  '\n']);
fprintf(file1,'.print tran  ');
fprintf(file1,'.save ');

for j=1:NOC
    fprintf(file1, ['I(' ['vl' 'r'  num2str(0) 'c' num2str(j) ] ')' '  ']);
end
fprintf(file1,'\n');

fprintf(file1,'.END');
fclose(file1);
end
