function [ Cell ] = GenerateCell(r, c, Gm)

rwl = [['R' 'r' num2str(r) 'c'  num2str(c)] ' ' ['r' num2str(r) 'c'  num2str(c-1)] ' ' ['r' num2str(r) 'c' num2str(c)] ' ' 'rwl' '\n'];
rbl = [['R' 'c' num2str(c) 'r'  num2str(r)] ' ' ['c' num2str(c) 'r'  num2str(r-1)] ' ' ['c' num2str(c) 'r' num2str(r)] ' ' 'rbl' '\n'];

Xmem= [['Rmr' num2str(r) 'c'  num2str(c)] ' ' ['r' num2str(r) 'c'  num2str(c)] ' ' ['c' num2str(c) 'r' num2str(r)] ' ' num2str(1/Gm) '\n'];

%% combine right and left columns
Cell = [rwl rbl Xmem];

end
