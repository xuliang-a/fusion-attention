require 'nn'
require 'nngraph'

local CONCAT = {}

function CONCAT.concat(rnn_size, output_size, dropout)
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  local outputs = {}
  -- 给出符号序列的索引
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  table.insert(inputs, nn.Identity()())
  local xf = inputs[1]
  local top_h_f = xf
  if dropout > 0 then top_h_f = nn.Dropout(dropout)(top_h_f) end
  local proj_f = nn.Linear(rnn_size, output_size)(top_h_f)

  local xb = inputs[2]
  local top_h_b = xb
  if dropout > 0 then top_h_b = nn.Dropout(dropout)(top_h_b) end
  local proj_b = nn.Linear(rnn_size, output_size)(top_h_b)

  local proj = nn.CAddTable()({proj_f, proj_b})
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return CONCAT

