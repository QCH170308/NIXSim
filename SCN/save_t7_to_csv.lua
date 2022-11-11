local thcsv=require 'thcsv'
require 'cunn'

if #arg<2 then error('Arguments: t7_filename csv_targetdir') end
print('Loading '..arg[1])
local net=torch.load(arg[1])
local params = net:parameters()
--print(params)

local cmul_cnt=0
local conv_cnt=0
local bias_cnt=0
for k,v in pairs(params) do
    if v:dim() == 2 then --CMul parameter
        cmul_cnt = cmul_cnt+1
        thcsv.write(arg[2]..'/CMul'..cmul_cnt..'.csv',v)
    elseif v:dim() == 3 then --WPM CMul parameter
        cmul_cnt = cmul_cnt+1
        thcsv.write(arg[2]..'/CMul'..cmul_cnt..'.csv',v:view(-1,v:size(3)))
    elseif v:dim() == 4 then --Conv weights
        conv_cnt = conv_cnt+1
        thcsv.write(arg[2]..'/Wght'..conv_cnt..'.csv',v:view(v:size(1),-1))
    elseif v:dim() == 1 then --Conv biases
        bias_cnt = bias_cnt+1
        thcsv.write(arg[2]..'/Bias'..bias_cnt..'.csv',v:view(v:size(1),1))
    else
        error('Unknown paramter dimension: '..v:dim())
    end
end

print('Done!')
