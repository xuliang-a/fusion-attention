local utils = require 'utils.utils'
local net_utils = {}


function net_utils.transform_cnn_conv(nDim)

  local cnn_part = nn.Sequential()

  cnn_part:add(nn.View(nDim, -1):setNumInputDims(3))
  cnn_part:add(nn.Transpose({2,3}))
  return cnn_part
end





function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end
function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  local count = {}
  for i=1,N do
    local tmp = 0
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      tmp = tmp + 1
      txt = txt .. word
    end
    --txt = txt .. '.'
    table.insert(count, tmp)

    table.insert(out, txt)
  end
  return out, count
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

function net_utils.clone_list_all(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    local new_sub = {}
    for m,n in pairs(v) do
      new_sub[m] = n:clone()
    end
    new[k] = new_sub
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, opt)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local id = utils.getopt(opt, 'id', 'evalscript')
  print(id)
  print(id)
  print(id)
  local dataset = utils.getopt(opt, 'dataset','captions_val2014')

  local out_struct = {val_predictions = predictions}
  print('../coco-caption/val' .. id .. '.json')
  utils.write_json('../coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  --print('../call_python_caption_eval.sh val' .. id .. '.json annotations/' ..dataset..'.json')
  local val_json = 'val' .. id .. '.json '
  local dataset_json = 'annotations/' ..dataset..'.json'
  print('val_json:',val_json)
  --os.execute('../call_python_caption_eval.sh '..val_json)
  --os.execute('../call_python_caption_eval.sh '..'val' .. id .. '.json')
  --os.execute('../call_python_caption_eval.sh '..val_json..dataset_json)
  print('../coco-caption/val' .. id .. '.json_out.json')

  local result_struct = utils.read_json('../coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  --local result_struct = "  "
  return result_struct
end

function net_utils.init_noise(graph, batch_size)
  if batch_size == nil then
    error('please provide valid batch_size value')
  end
  for i, node in pairs(graph:listModules()) do
    local layer = graph:get(i)
    local t = torch.type(layer)
    if t == 'nn.DropoutFix' then
      layer:init_noise(batch_size)
    end
  end
end

function net_utils.deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = net_utils.deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function net_utils.setBNGradient0(graph)
   -- setting the gradient of BN to be zero
  local BNlayers = graph:findModules('nn.SpatialBatchNormalization')
  for i, node in pairs(BNlayers) do
    node.gradWeight:zero()
    node.gradBias:zero()
  end
end



-- takes a batch of images and preprocesses them
-- VGG-16 network is hardcoded, as is 224 as size to forward
function net_utils.prepro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 224

  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then 
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end

  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  if not net_utils.vgg_mean then
    net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  end
  net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match

  -- subtract vgg mean
  imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))

  return imgs
end

return net_utils