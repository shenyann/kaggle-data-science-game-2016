local initialize = {}

function initialize.MSRinit(model)
    for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
        local n = v.kW*v.kH*v.nOutputPlane
        v.weight:normal(0,math.sqrt(2/n))
        if v.bias then v.bias:zero() end
    end
end

function initialize.FCinit(model)
    for k,v in pairs(model:findModules'nn.Linear') do
        v.bias:zero()
    end
end

function initialize.DisableBias(model)
    for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
        v.bias = nil
        v.gradBias = nil
    end
end
function initialize.makeDataParallelTable(model, nGPU)
    if nGPU > 1 then
        local gpus = torch.range(1, nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpus)
        :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
        end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end
    return model
end

return initialize
