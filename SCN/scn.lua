require 'sys'
require 'optim'
require 'cunn'

sys.tic()

local cmd = torch.CmdLine()
cmd:option('-numLayers', 3, 'Number of layers including first and last layers')
cmd:option('-hiddenChannels', 4, 'Hidden channel size')
cmd:option('-maxEpochs', 100, 'Maximum epochs')
cmd:option('-displayInterval', 10, 'Display loss every n epochs')
cmd:option('-saveInterval', 100, 'Save model every n epochs')
cmd:option('-saveName', '', 'Model savefile name prefix')
cmd:option('-learningRate', 1e-2, 'Learning rate')
cmd:option('-batchSize', 50, 'Batch size')
cmd:option('-optimizer', 'sgd', 'Optimizer (sgd or adam)')
cmd:option('-load', '', 'Load pretrained network')
cmd:option('-testOnly', false, 'Run test only')
cmd:option('-mask', '', 'Mask file')
cmd:option('-mask2', '', 'Mask file')
cmd:option('-trainData', 'W50000.t7', 'Train data')
cmd:option('-testData', 'W10000.t7', 'Test data')
cmd:option('-cbsize', 128, 'Crossbar size')
local opt = cmd:parse(arg)

--[[
Data = {
    input = (N, H, W)
    target = (N, H, W)
    size = N
}
--]]
local trainData = torch.load(opt.trainData)
local testData = torch.load(opt.testData)


function dictCuda(dict, keys)
    for _, key in pairs(keys) do
        dict[key] = dict[key]:cuda()
    end
end

function applyMask(X, mask)
    for i=1,X:size(1) do
        X[i]:cmul(mask)
    end
end

function applyMask2(X, mask2p, mask2n)
    local Xn = X:clone()
    X[X:lt(0)] = 0
    Xn[Xn:gt(0)] = 0
    for i=1,X:size(1) do
        X[i]:cmul(mask2p):csub(Xn[i]:cmul(mask2n))
    end
end

local shape = trainData.input[1]:size()
--local row, col = 128, 128
function readMask(maskfile)
    local maskmatrix = torch.CudaTensor(shape):fill(1)
    local i = 0
    for line in io.lines(maskfile) do
        i = i+1
        maskmatrix:select(1, i):copy(torch.Tensor(string.split(line, ",")))
    end
    assert(i == shape[1])
    return maskmatrix
end

function readMask2(maskfile)
    local maskmatrix2p = torch.CudaTensor(shape):fill(1)
    local maskmatrix2n = torch.CudaTensor(shape):fill(1)
    local i = 0
    for line in io.lines(maskfile) do
        i = i+1
        if ( i <= shape[1]) then
            maskmatrix2p:select(1, i):copy(torch.Tensor(string.split(line, ",")))
        else
            maskmatrix2n:select(1, i - shape[1]):copy(torch.Tensor(string.split(line, ",")))
        end
    end
    assert(i == shape[1] * 2)
    return maskmatrix2p, maskmatrix2n
end

local keys = {'input','target'}
dictCuda(trainData, keys)
dictCuda(testData, keys)

if opt.mask ~= '' then
    local mask = readMask(opt.mask)
    applyMask(trainData.input, mask)
    applyMask(testData.input, mask)
end

if opt.mask2 ~= '' then
    local mask2p, mask2n = readMask2(opt.mask2)
    applyMask2(trainData.input, mask2p, mask2n)
    applyMask2(testData.input, mask2p, mask2n)
end

--[[
local net = nn.Sequential()
net:add(nn.View(1,128,128))
net:add(nn.SpatialConvolution(1,32,3,3,1,1,1,1))
net:add(nn.ReLU())
--net:add(nn.SpatialMaxPooling())
net:add(nn.SpatialConvolution(32,32,3,3,1,1,1,1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(32,1,3,3,1,1,1,1))
net:cuda()
--]]

local net
if opt.load ~= '' then
    net = torch.load(opt.load)
else
    local num_layers = opt.numLayers  --including first/last layer
    local hidden_channels = opt.hiddenChannels

    net = nn.Sequential()
    net:add(nn.CMul(opt.cbsize,opt.cbsize))
    net:add(nn.View(1,opt.cbsize,opt.cbsize))
    net:add(nn.SpatialConvolution(1, hidden_channels, 3, 3, 1, 1, 1, 1))
    net:add(nn.ReLU())
    for i=1,num_layers-2 do
        net:add(nn.SpatialConvolution(hidden_channels, hidden_channels, 3, 3, 1, 1, 1, 1))
        net:add(nn.ReLU())
    end
    net:add(nn.SpatialConvolution(hidden_channels, 1, 3, 3, 1, 1, 1, 1))
    net:add(nn.View(opt.cbsize,opt.cbsize))
    net:add(nn.CMul(opt.cbsize,opt.cbsize))
    net:cuda()
end

print(net)

local criterion = nn.MSECriterion()
criterion:cuda()

sgd_params = {
    learningRate = opt.learningRate,
--    learningRateDecay = 1e-4,
--    weightDecay = 1e-3,
--    momentum = 1e-4
}
adam_params = {
    learningRate = opt.learningRate
}

parameters, gradParameters = net:getParameters()

function train(dataset)
    local current_loss = 0
    local shuffle = torch.randperm(dataset.size)
    local batch_size = opt.batchSize
    
    for t = 0, dataset.size-1, batch_size do
        local size = math.min(t + batch_size, dataset.size) - t
        local inputs = torch.CudaTensor(size, opt.cbsize, opt.cbsize)
        local targets = torch.CudaTensor(size, opt.cbsize, opt.cbsize)
        for i = 1,size do
            local input = dataset.input[shuffle[i+t]]
            local target = dataset.target[shuffle[i+t]]
            inputs[i] = input
            targets[i] = target
        end

        local feval = function(x)
            if parameters ~= x then parameters:copy(x) end
            gradParameters:zero()

            local batch_loss = criterion:forward(net:forward(inputs), targets)
            net:backward(inputs, criterion:backward(net.output, targets))

            return batch_loss, gradParameters
        end

        if opt.optimizer == 'sgd' then
            _, fs = optim.sgd(feval, parameters, sgd_params)
        elseif opt.optimizer == 'adam' then
            _, fs = optim.adam(feval, parameters, adam_params)
        else
            print(opt.optimizer .. " optimizer is not supported. Use sgd instead")
            _, fs = optim.sgd(feval, parameters, sgd_params)
        end

        current_loss = current_loss + fs[1]*size
    end
    
    return current_loss / dataset.size
end

function test(dataset)
    local current_loss = 0
    local batch_size = opt.batchSize

    for t = 0, dataset.size-1, batch_size do
        local size = math.min(t + batch_size, dataset.size) - t
        local inputs = dataset.input:sub(t+1, t+size)
        local targets = dataset.target:sub(t+1, t+size)
        local batch_loss = criterion:forward(net:forward(inputs), targets)
        current_loss = current_loss + batch_loss*size
    end

    return current_loss / dataset.size
end

local train_loss
local test_loss
        
local max_epoch = opt.maxEpochs
local display_interval = opt.displayInterval
local save_interval = opt.saveInterval
local prefix = opt.saveName
if prefix ~= '' then prefix = prefix .. '_' end
net.epoch = net.epoch or 0

if opt.testOnly then
    train_loss = test(trainData)
    test_loss = test(testData)
    print(string.format("%.2fm - Epoch %d, train_loss %.4e, test_loss %.4e",
                        sys.toc()/60, net.epoch, train_loss, test_loss))
else
    while net.epoch < max_epoch do
        net.epoch = net.epoch + 1
        train_loss = train(trainData)
        test_loss = test(testData)

        if net.epoch % display_interval == 0 then
            print(string.format("%.2fm - Epoch %d, train_loss %.4e, test_loss %.4e",
                                sys.toc()/60, net.epoch, train_loss, test_loss))
        end
        if net.epoch % save_interval == 0 then torch.save(prefix..'epoch'..net.epoch..'.model', net) end
    end
end
