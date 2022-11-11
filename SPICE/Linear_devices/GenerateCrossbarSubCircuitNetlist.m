function SubcitName=GenerateCrossbarSubCircuitNetlist(Gm,NOR,NOC,r)
%Generates a netlist of crossbar subcircuit SPICE simulation
[temp1, rand_string] = fileparts(tempname);
SubcitName=['sub_', rand_string];
%% properties
fileName = [SubcitName '.sub'];

rwl = r;
rbl = r;

%% File generation
file = fopen(fileName, 'w');

%% Write circuit description to the file
fprintf(file, '* Spice netlist file for Resistive crossbar subcircuit\n\n');
fprintf(file,[ '.SUBCKT  ' SubcitName '  '] );

for r=1:1:NOR
    fprintf(file,[ 'r' num2str(r) 'c'  num2str(0) ' ']);
end
for r=1:1:NOR
    fprintf(file,[ 'r' num2str(r) 'c'  num2str(NOC) ' ']);
end

for c=1:1:NOC
    fprintf(file, ['c'  num2str(c) 'r' num2str(0) ' ']);
end
for c=1:1:NOC
    fprintf(file, ['c'  num2str(c) 'r' num2str(NOR) ' ']);
end

fprintf(file, '\n');


fprintf(file, ['.param rwl=' num2str(rwl) '\n']);
fprintf(file, ['.param rbl=' num2str(rbl) '\n']);
fprintf(file, '\n');

% Crossbar Cells
for i=1:NOR
    fprintf(file, ['*** Row '  num2str(i) ' ***\n']);

    for j=1:NOC
        fprintf(file, ['*** Column '  num2str(j) ' ***\n']);
        CrossbarCell = GenerateCell(i, j, Gm(i,j));
        fprintf(file, CrossbarCell);
    end
    fprintf(file,'\n');
end

%% End of the file
fprintf(file,[ '.ENDS' '  ' 'CrossbarSubCir  \n']);

%% Close opened file
fclose(file);

end

