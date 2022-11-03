--returns a table which has 'x' layers of 'y' nodes initialized with random weights and biases between 0 and 1
--x is simply the numbers of layers inputted
--y is the value of the number/s inputted.
--example: makeNet(1,5,8,3) would return a table of 4 layers with the following node counts:
-- layer 1: 1 node
-- layer 2: 5 nodes
-- layer 3: 8 nodes
-- layer 4: 3 nodes
local function makeNet(...)

    local layers = {...}
    local net = {}

    for layer = 2, #layers do

        net[layer-1] = {}

        for node = 1, layers[layer] do

            net[layer-1][node] = {
                b = math.random()-0.5,
                costb = 0,
            }

            for connection = 1, layers[layer-1] do
                net[layer-1][node][connection] = {
                    w = math.random()-0.5,
                    costw = 0,
                }
            end
        end
    end
    return net
end
--[[
Example structure:
local net = makeNet(2,3,2)

(2,3,2 means 2 inputs, 1 hidden layeer of 3 nodes, and an output layer of 2 nodes)
net = {
      (hidden layer 1, 3 nodes, 2 connections each because previous layer has 2 nodes) 
      (each node has a bias and a bias cost, and each connection inside that node has a weight and a weight cost)
    {
          (node 1) 
        {
            b,costb,
              (connection 1) 
            { w,costw },
              (connection 2) 
            {..},
        },
          (node 2) 
        {
            b,costb,
            {..},
            {..},
        }
          (node 3) 
        {
            b,costb,
            {..},
            {..},
        }
    },
      (output layer, 2 nodes, 3 connections each because previous layer has 3 nodes)
    {
        {
            b,costb,
            {..},
            {..},
        },
        {
            b,costb,
            {..},
            {..},
        }
    }
}

]]

--sigmoid function for node activation
local function sigmoidActivation(x)
    return 1/(1+2.71828^(-x))
end

--evaluate node cost derivative: cost/activation
local function nodeCostDerivative(activation, expectedOutput)
    return 2*(activation-expectedOutput)
end

--evaluate activation derivative: activation/weightedInput
local function nodeActivationDerivative(weightedInput)
    return weightedInput*(1-weightedInput)
end

--returns the outputs of every node in a given net for inputs
local function getNetOutputs(inputs,net)
    --run inputs through the whole net
    local lastInputs = inputs

    for layer = 1, #net do

        local layerOutput = {}
        for node = 1, #net[layer] do

            local sum = net[layer][node].b
            for connection = 1, #net[layer][node] do

                sum = sum + net[node][connection].w*lastInputs[node]

            end
            layerOutput[node] = sigmoidActivation(sum)

        end
        lastInputs = layerOutput

    end
    return lastInputs
end

--returns the average cost value of a net for a given set of data
local function getAvgNetCost(inputs,expectedOutputs,net)
    local sum = 0
    local outputs = getNetOutputs(inputs,net)
    for dataPoint = 1, #inputs do
        local error = (outputs[dataPoint] - expectedOutputs[dataPoint])
        sum = sum + (error * error)
    end
    return sum / #outputs
end

--Applies cost gradients to net
local function applyGradients(learnRate,net)
    for layerLookAt = 1, #net do
        local currentNet = net[layerLookAt]
        for nodeLookAt = 1, #currentNet do
            local currentNode = currentNet[nodeLookAt]
            net[layerLookAt][nodeLookAt].b = currentNode.b - currentNode.costb*learnRate
            net[layerLookAt][nodeLookAt].w = currentNode.b - currentNode.costw*learnRate
        end
    end
end

--trains the net using backpropagation
local function trainNet(inputs,expectedOutputs,net,learnRate)

    local h = 0.0001
    local originalCost = getAvgNetCost(inputs,expectedOutputs,net)

    for layer = 1,  #net do
        for node = 1, #net[layer] do
            for connection = 1, #net[layer][node] do
                --calculate the cost gradient for the current weights
                net[layer][node][connection].w = net[layer][node][connection].w + h
                local deltaCost = getAvgNetCost(inputs,expectedOutputs,net) - originalCost
                net[layer][node][connection].w = net[layer][node][connection].w - h
                net[layer][node][connection].costw = deltaCost / h
            end

            --calculate the cost gradient for the current biases
            net[layer][node].b = net[layer][node].b + h
            local deltaCost = getAvgNetCost(inputs,expectedOutputs,net) - originalCost
            net[layer][node].b = net[layer][node].b - h
            net[layer][node].costb = deltaCost / h
        end
    end
    applyGradients(learnRate,net)
end

return {
    makeNet = makeNet,
    getNetOutputs = getNetOutputs,
    getAvgNetCost = getAvgNetCost,
    trainNet = trainNet
}
