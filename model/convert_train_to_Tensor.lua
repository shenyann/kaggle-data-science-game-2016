sys=require 'sys'
ffi=require 'ffi'
require 'image'
trainFile=io.open('train.txt')
trainPath={}
trainLabel={}
train = {}
while true do
    local tmp={}
    line=trainFile:read('*line')
    if not line then break end
    for i in string.gmatch(line,'%S+') do
        table.insert(tmp,i)

    end
    table.insert(trainPath,tmp[1])
    table.insert(trainLabel,tonumber(tmp[2]))   
    collectgarbage()
end
train.data = torch.FloatTensor(#trainPath,3,32,32)
for i=1, #trainPath do
    img = image.load(trainPath[i],3,'float')
    img = image.scale(img,32,32,'bicubic')
    train.data[i]:copy(img)
end
train.label=torch.FloatTensor(trainLabel)
torch.save('data.t7', train)





