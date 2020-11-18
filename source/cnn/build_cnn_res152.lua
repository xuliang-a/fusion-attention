local utils = require 'utils.utils'
local build_cnn_res152 = {}

function build_cnn_res152.conv1_to_conv2x(cnn, opt)
  local layer_num = utils.getopt(opt, 'start_layer_num', 6)
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

  local cnn_part = nn.Sequential()
  for i = 1, layer_num-1 do
    local layer = cnn:get(i)

    cnn_part:add(layer)
  end

  return cnn_part
end


function build_cnn_res152.conv3x_to_conv5x(cnn, opt)
  local start_layer_num = utils.getopt(opt, 'start_layer_num', 6)
  local layer_num = utils.getopt(opt, 'layer_num', 8)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  local cnn_part = nn.Sequential()
  for i = start_layer_num, layer_num do
    local layer = cnn:get(i)

    cnn_part:add(layer)
  end

  return cnn_part
end


function build_cnn_res152.full_conn_2048(cnn, opt)
  local layer_num_start = utils.getopt(opt, 'layer_num', 9)
  local layer_num_end = utils.getopt(opt, 'layer_num', 10)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  
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
  for i = layer_num_start, layer_num_end do
    local layer = cnn:get(i)
    
    cnn_part:add(layer)
  end

  return cnn_part
end

return build_cnn_res152