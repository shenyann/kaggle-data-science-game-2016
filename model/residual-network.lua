local nn = require 'nn'
local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local initialize = paths.dofile'initialize_weight.lua'
local function Dropout()
    return nn.Dropout(opt.dropout,nil,true)
end
local function createModel(opt)
    local depth = opt.depth
    local blocks = {}
    local function basic(nInputPlane, nOutputPlane, stride)
        local convolutional_params = {
            {3,3,stride,stride,1,1},
            {3,3,1,1,1,1},
        }
        local nBasicblockPlane = nOutputPlane
        local block = nn.Sequential()
        local convs = nn.Sequential()     
        for i,v in ipairs(convolutional_params) do
            if i == 1 then
                local module = nInputPlane == nOutputPlane and convs or block
                module:add(SBatchNorm(nInputPlane)):add(ReLU(true))
                convs:add(Convolution(nInputPlane,nBasicblockPlane,table.unpack(v)))
            else
                convs:add(SBatchNorm(nBasicblockPlane)):add(ReLU(true))
                if opt.dropout > 0 then
                    convs:add(Dropout())
                end
                convs:add(Convolution(nBasicblockPlane,nBasicblockPlane,table.unpack(v)))
            end
        end
        local shortcut = nInputPlane == nOutputPlane and
        nn.Identity() or Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)

        return block
        :add(nn.ConcatTable()
        :add(convs)
        :add(shortcut))
        :add(nn.CAddTable(true))
    end

    local function layer(block, nInputPlane, nOutputPlane, count, stride)
        local s = nn.Sequential()
        s:add(block(nInputPlane, nOutputPlane, stride))
        for i=2,count do
            s:add(block(nOutputPlane, nOutputPlane, 1))
        end
        return s
    end

    local model = nn.Sequential()
    do
        assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
        local n = (depth - 4) / 6
        local k = opt.widen_factor
        local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}
        model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) 
        model:add(layer(basic,nStages[1],nStages[2],n,1)) 
        model:add(layer(basic,nStages[2],nStages[3],n,2))
        model:add(layer(basic,nStages[3],nStages[4],n,2))
        model:add(SBatchNorm(nStages[4]))                        
        model:add(ReLU(true))
        model:add(Avg(8, 8, 1, 1))
        model:add(nn.View(nStages[4]):setNumInputDims(3))
        model:add(nn.Linear(nStages[4], opt.num_classes))
    end
    initialize.DisableBias(model)
    initialize.MSRinit(model)
    initialize.FCinit(model)
    return model
end

return createModel(opt)
