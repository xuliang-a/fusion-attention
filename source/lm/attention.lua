require 'nn'
require 'nngraph'

local attention = {}
function attention.attention(input_size, rnn_size, output_size, dropout)
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- top_h
  table.insert(inputs, nn.Identity()()) -- fake_region
  table.insert(inputs, nn.Identity()()) -- conv_feat
  table.insert(inputs, nn.Identity()()) -- conv_feat_embed
  table.insert(inputs, nn.Identity()()) -- prev_h


  local h_out = inputs[1]
  local fake_region = inputs[2]
  local conv_feat = inputs[3]
  local conv_feat_embed = inputs[4]
  local prev_h = inputs[5]

  local fake_region = nn.ReLU()(nn.Linear(rnn_size, input_size)(fake_region))
  -- view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
  if dropout > 0 then fake_region = nn.Dropout(dropout)(fake_region) end
  -- wcy:: fake_region_embed(5,512)
  local fake_region_embed = nn.Linear(input_size, input_size)(fake_region)

  local h_out_linear = nn.Tanh()(nn.Linear(rnn_size, input_size)(h_out))
  if dropout > 0 then h_out_linear = nn.Dropout(dropout)(h_out_linear) end
  -- wcy:: h_out_embed(5,512)
  local h_out_embed = nn.Linear(input_size, input_size)(h_out_linear)


  local prev_h_linear = nn.Tanh()(nn.Linear(rnn_size, input_size)(prev_h))
  if dropout > 0 then prev_h_linear = nn.Dropout(dropout)(prev_h_linear) end
  -- wcy:: prev_h_embed(5,512)
  local prev_h_embed = nn.Linear(input_size, input_size)(prev_h_linear)


  -- wcy:: txt_replicate(5，49,512)
  local txt_replicate_prev = nn.Replicate(49,2)(prev_h_embed)
  -- wcy:: txt_replicate(5，50,512)
  local txt_replicate = nn.JoinTable(2)({nn.View(-1,1,input_size)(h_out_embed), txt_replicate_prev})
  -- nn.View(-1,1,input_size)(fake_region) -->(5,1,512)
  -- wcy:: img_all(5,50,512)
  local img_all = nn.JoinTable(2)({nn.View(-1,1,input_size)(fake_region), conv_feat})
  -- wcy:: img_all_embed(5,50,512)
  local img_all_embed = nn.JoinTable(2)({nn.View(-1,1,input_size)(fake_region_embed), conv_feat_embed})

  -- wcy:: (5,50,512)
  local hA = nn.Tanh()(nn.CAddTable()({img_all_embed, txt_replicate}))
  if dropout > 0 then hA = nn.Dropout(dropout)(hA) end
  -- wcy::nn.View(input_size):setNumInputDims(2)(hA) -->(250,512)
  -- wcy::hAflat维度(250,1)
  local hAflat = nn.Linear(input_size,1)(nn.View(input_size):setNumInputDims(2)(hA))
  -- wcy::nn.View(50):setNumInputDims(2)(hAflat)  --> (5,50)
  -- wcy::PI是概率值 （各区域特征向量和st每一个都有一个概率值）
  local PI = nn.SoftMax()(nn.View(50):setNumInputDims(2)(hAflat))

  -- wcy:: (5,1,50)
  local probs3dim = nn.View(1,-1):setNumInputDims(1)(PI)
  -- wcy:: (5,1,512)
  local visAtt = nn.MM(false, false)({probs3dim, img_all})
  -- wcy:: 最终的自适应上下文向量(5,512)
  local visAttdim = nn.View(input_size):setNumInputDims(2)(visAtt)

  local atten_out = nn.CAddTable()({visAttdim, h_out_linear})

  local h = nn.Tanh()(nn.Linear(input_size, input_size)(atten_out))
  if dropout > 0 then h = nn.Dropout(dropout)(h) end
  local proj = nn.Linear(input_size, output_size)(h)

  local logsoft = nn.LogSoftMax()(proj)
  --local logsoft = nn.SoftMax()(proj)
  -- wcy:: (5, self.vocab_size+1)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end
return attention