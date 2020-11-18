require 'nn'
require 'nngraph'

local attention = {}
function attention.attention(input_size, rnn_size, output_size, dropout)
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- top_h
  table.insert(inputs, nn.Identity()()) -- fake_region
  table.insert(inputs, nn.Identity()()) -- fc_feat

  local h_out = inputs[1]
  local fake_region = inputs[2]
  local fc_feat = inputs[3]


  local fc_feat_embed = nn.Linear(input_size, input_size)(fc_feat)

  local fake_region = nn.ReLU(true)(nn.Linear(rnn_size, input_size)(fake_region))
  -- view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
  if dropout > 0 then fake_region = nn.Dropout(dropout)(fake_region) end
  
  local fake_region_embed = nn.Linear(input_size, input_size)(fake_region)

  local h_out_linear = nn.Tanh()(nn.Linear(rnn_size, input_size)(h_out))
  if dropout > 0 then h_out_linear = nn.Dropout(dropout)(h_out_linear) end

  local h_out_embed = nn.Linear(input_size, input_size)(h_out_linear)

  local txt_replicate = nn.Replicate(2,2)(h_out_embed)

  local img_all = nn.JoinTable(2)({nn.View(-1,1,input_size)(fake_region), nn.View(-1,1,input_size)(fc_feat)})
  local img_all_embed = nn.JoinTable(2)({nn.View(-1,1,input_size)(fake_region_embed), nn.View(-1,1,input_size)(fc_feat_embed)})

  local hA = nn.Tanh()(nn.CAddTable()({img_all_embed, txt_replicate}))
  if dropout > 0 then hA = nn.Dropout(dropout)(hA) end
  local hAflat = nn.Linear(input_size,1)(nn.View(input_size):setNumInputDims(2)(hA))  
  local PI = nn.SoftMax()(nn.View(2):setNumInputDims(2)(hAflat))

  local probs3dim = nn.View(1,-1):setNumInputDims(1)(PI)
  local visAtt = nn.MM(false, false)({probs3dim, img_all})
  local visAttdim = nn.View(input_size):setNumInputDims(2)(visAtt)
  local atten_out = nn.CAddTable(2)({visAttdim, h_out_linear})

 -- local h = nn.Tanh()(nn.Linear(input_size, input_size)(atten_out))
  if dropout > 0 then atten_out = nn.Dropout(dropout)(atten_out) end
  local proj = nn.Linear(input_size, output_size)(atten_out)
  -- local proj = nn.Linear(input_size*2, output_size)(atten_out)

  local logsoft = nn.LogSoftMax()(proj)
  --local logsoft = nn.SoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end
return attention