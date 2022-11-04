--Prints out the contents of the given net. 
--See the comment below makeNet() for an example.
local function netDebug(net)
    print("\nPrinting out net...")
    print("L = layer, N = node\n")
    local layerSum = #net
    local nodeSum = 0
    local connectionSum = 0
    for layer = 1, #net do
        print("L"..layer..":")
        for node = 1, #net[layer] do
            nodeSum = nodeSum + 1
            print("    N"..node..", B="..(net[layer][node].b).." : "..#net[layer][node].." connections")
            for connection = 1, #net[layer][node] do
                print("            W"..connection..": "..(net[layer][node][connection].w))
                connectionSum = connectionSum + 1
            end
        end
    end
    print("\nTotals: "..layerSum.." layers, "..nodeSum.." nodes, "..connectionSum.." connections.")
    print("Press Enter to continue.")
    read()
end

--returns a 3d table of the created net
--input numbers. Each number creates a new layer with nodes equal to the inputted number
--each node has connections equal to the previous layers node count (example below)
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

L = layer, N = node
L1:
    N1: 2 connections
        W1: __
        W2: __
    N2: 2 connections
        ...
    N3: 2 connections
        ...
L2:
    N1: 3 connections
        ...
    N2: 3 connections
        ...

Totals: 2 layers, 5 nodes, 12 connections

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

                sum = sum + net[layer][node][connection].w*lastInputs[connection]

            end
            layerOutput[node] = sigmoidActivation(sum)

        end
        lastInputs = layerOutput

    end
    return lastInputs
end

--returns the average cost value of a net for a given set of data
local function getAvgNetCost(inputs,expectedOutputs,net)
    local avgSum = 0
    for set = 1, #expectedOutputs do
        local sum = 0
        local outputs = getNetOutputs(inputs[set],net)
        for dataPoint = 1, #expectedOutputs[set] do
            local error = (outputs[dataPoint] - expectedOutputs[set][dataPoint])
            sum = sum + (error * error)
        end
        avgSum = avgSum + (sum/#outputs)
    end
    return avgSum / #expectedOutputs[1]
end

--Applies cost gradients to net
local function applyGradients(learnRate,net)
    for layer = 1, #net do
        for node = 1, #net[layer] do

            net[layer][node].b = net[layer][node].b - net[layer][node].costb*learnRate

            for connection = 1, #net[layer][node] do

                net[layer][node][connection].w = net[layer][node][connection].w - net[layer][node][connection].costw*learnRate
            
            end
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
    trainNet = trainNet,
    netDebug = netDebug
}
