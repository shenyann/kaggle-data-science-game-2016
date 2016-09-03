require 'xlua'
require 'optim'
require 'image'
require 'cunn'
require 'cudnn'
local c = require 'trepl.colorize'
local json = require 'cjson'
paths.dofile'augmentation.lua'
local optnet = require 'optnet'
local iterm = require 'iterm'
require 'iterm.dot'
local utils = paths.dofile'initialize_weight.lua'

opt = {
  dataset = 'data32x32.t7',
  save = 'logs',
  batchSize = 128,
  learningRate = 0.1,
  learningRateDecay = 0,
  learningRateDecayRatio = 0.2,
  weightDecay = 0.0005,
  dampening = 0,
  momentum = 0.9,
  epoch_step = "80",
  max_epoch = 300,
  model = 'residual-network',
  optimMethod = 'sgd',
  init_value = 10,
  depth = 28,
  nesterov = true,
  dropout = 0.3,
  hflip = true,
  randomcrop = 4,
  imageSize = 32,
  randomcrop_type = 'reflection',
  cudnn_fastest = true,
  cudnn_deterministic = true,
  optnet_optimize = true,
  multiply_input_factor = 1,
  widen_factor = 2,
  nGPU = 2,
}
opt = xlua.envparams(opt)
opt.epoch_step = tonumber(opt.epoch_step) or loadstring('return '..opt.epoch_step)()
print(opt)
print(c.blue '==>' ..' loading data')
local provider = torch.load(opt.dataset)
opt.num_classes = provider.label:max()
print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
local net = dofile(opt.model..'.lua'):cuda()
do
   function nn.Copy.updateGradInput() end
   local function add(flag, module) if flag then model:add(module) end end
   add(opt.hflip, nn.BatchFlip():float())
   add(opt.randomcrop > 0, nn.RandomCrop(opt.randomcrop, opt.randomcrop_type):float())
   model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
   add(opt.multiply_input_factor ~= 1, nn.MulConstant(opt.multiply_input_factor):cuda())
   cudnn.convert(net, cudnn)
   cudnn.benchmark = true
   if opt.cudnn_fastest then
      for i,v in ipairs(net:findModules'cudnn.SpatialConvolution') do v:fastest() end
   end
   if opt.cudnn_deterministic then
      net:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
   end
   print(net)
   print('Network has', #net:findModules'cudnn.SpatialConvolution', 'convolutions')
   local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
   if opt.optnet_optimize then
      optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
   end
   model:add(utils.makeDataParallelTable(net, opt.nGPU))
end
local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end
print('Will save at '..opt.save)
paths.mkdir(opt.save)
local parameters,gradParameters = model:getParameters()
opt.n_parameters = parameters:numel()
print('Network has ', parameters:numel(), 'parameters')
print(c.blue'==>' ..' setting criterion')
local criterion = nn.CrossEntropyCriterion():cuda()
local f = function(inputs, targets)
   model:forward(inputs)
   local loss = criterion:forward(model.output, targets)
   local df_do = criterion:backward(model.output, targets)
   model:backward(inputs, df_do)
   return loss
end
print(c.blue'==>' ..' configuring optimizer')
local optimState = tablex.deepcopy(opt)
function train()
  model:training()

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.data:size(1)):long():split(opt.batchSize)
  indices[#indices] = nil
  local loss = 0
  for t,v in ipairs(indices) do
    local inputs = provider.data:index(1,v)
    targets:copy(provider.label:index(1,v))
    optim[opt.optimMethod](function(x)
      if x ~= parameters then parameters:copy(x) end
      model:zeroGradParameters()
      loss = loss + f(inputs, targets)
      return f,gradParameters
    end, parameters, optimState)
  end
  return loss / #indices
end
function test()
  model:evaluate()
  local confusion = optim.ConfusionMatrix(opt.num_classes)
  local data_split = provider.data:split(opt.batchSize,1)
  local labels_split = provider.label:split(opt.batchSize,1)
  for i,v in ipairs(data_split) do
    confusion:batchAdd(model:forward(v), labels_split[i])
  end
  confusion:updateValids()
  return confusion.totalValid * 100
end
for epoch=1,opt.max_epoch do
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  if torch.type(opt.epoch_step) == 'number' and epoch % opt.epoch_step == 0 or
     torch.type(opt.epoch_step) == 'table' and tablex.find(opt.epoch_step, epoch) then
    opt.learningRate = opt.learningRate * opt.learningRateDecayRatio
    optimState = tablex.deepcopy(opt)
  end
  local function t(f) local s = torch.Timer(); return f(), s:time().real end
  local loss, train_time = t(train)
  local test_acc, test_time = t(test)
  log{
     loss = loss,
     epoch = epoch,
     test_acc = test_acc,
     lr = opt.learningRate,
     train_time = train_time,
     test_time = test_time,
   }
end
torch.save(opt.save..'/model.t7', net:clearState())

