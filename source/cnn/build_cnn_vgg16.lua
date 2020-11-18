local utils = require 'utils.utils'
local build_cnn_vgg16 = {}



function build_cnn_vgg16.conv_layers(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 31)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)
    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end


    cnn_part:add(layer)
  end

  return cnn_part
end


function build_cnn_vgg16.full_conn_4096(cnn, opt)
  local layer_num_start = utils.getopt(opt, 'layer_num_start', 32)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = layer_num_start, layer_num do
    local layer = cnn:get(i)

    cnn_part:add(layer)
  end

  return cnn_part
end

    
return build_cnn_vgg16