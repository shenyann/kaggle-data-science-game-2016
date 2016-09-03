sys=require 'sys'
ffi=require 'ffi'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'image'
testFile=io.open('test.txt')
testPath={}
while true do
    local tmp={}
    line=testFile:read('*line')
    if not line then break end
    for i in string.gmatch(line,'%S+') do
        table.insert(tmp,i)

    end
    table.insert(testPath,tmp[1])
    --  table.insert(trainLabel,tonumber(tmp[2]))   --here is space
    collectgarbage()
end

local model = torch.load('wide-residual-networks/32model.t7')
local softmax = cudnn.SoftMax():cuda()
model:add(softmax)
model:evaluate()
print('Model: ...',model)


for i=1, #testPath do
    local img = image.load(testPath[i],3,'float')
    local img = image.scale(img,32,32,'bicubic')
    local batch = img:view(1,table.unpack(img:size():totable()))
    local output = model:forward(batch:cuda())
    local probs, index = output:topk(1, true, true)
    print(paths.basename(testPath[i]),index[1],probs)
end






