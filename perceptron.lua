--returns a 2d table of random (from -0.5 to 0.5) weights and biases
local function makeLayer(size)
    local layer = {}
    for i = 1, size do
        layer[i] = { 
            w=math.random()-0.5,
            b=math.random()-0.5
        }
    end
    return layer
end

--returns a 3d table neural network which has 'x' layers of 'y' neuron count of random weights and biases between 0 and 1, where x is simply the numbers of layers inputted, and y is the value of the numbers inputted.
--example: makeNeuralNet(1,5,8,3) would return a table of 4 layers with the following neuron counts:
-- layer 1: 1 neuron
-- layer 2: 5 neurons
-- layer 3: 8 neurons
-- layer 4: 3 neurons
local function makeNeuralNet(...)
    local neuronList = {}
    local args = {...}
    for i = 1, #args do
        neuronList[i] = makeLayer(args[i])
    end
    return neuronList
end

--sigmoid function for neuron activation
local function sigmoid(x)
    return 1/(1+2.718^(-x))
end

--step function for neuron activation
local function heavySide(x)
    if x <= 0 then
        return 0
    else
        return 1
    end
end

local activationFunction = sigmoid

--calculate cost of actual results from expected results (result - expected)^2
local function getResultDelta(results,expected)
    if #results ~= #expected then
        error("Error: Length of expected results does not match length or actual results.\nLength of 'expected': "..#expected.."\nLength of 'results': "..#results)
    end
    local delta = 0
    for i = 1, #results do
        delta = delta + (results[i] - expected[i])^2
    end
    return delta
end

--slightly changes some of the weights and biases by +/- 'weightStep' and 'biasStep' in a neural net
local function netScramble(net,weightStep,biasStep)
    local scrambledNet = {}
    for layer = 1, #net do
        scrambledNet[layer] = {}
        for neuron = 1, #net[layer] do
            scrambledNet[layer][neuron] = {}
            local randomSign1 = 1
            local randomSign2 = 1
            if math.random() < 0.5 then randomSign1 = -1 end
            if math.random() < 0.5 then randomSign2 = -1 end
            scrambledNet[layer][neuron].w = net[layer][neuron].w + weightStep*randomSign1
            scrambledNet[layer][neuron].b = net[layer][neuron].b + biasStep*randomSign2
        end
    end
    return scrambledNet
end

--returns 1d table of result values from 1d input table and 2d layer table of weights and biases
local function iterateLayer(inputs,layer)
    local results = {}
    for i = 1, #layer do
        local sum = 0
        for j = 1, #inputs do
            sum = sum + layer[i].w*inputs[j]
        end
        results[i] = activationFunction(sum+layer[i].b)
    end
    return results
end

--returns 1d results table from a neural net with 1d input table and 3d neural net table
local function iterateNet(inputs,net)
    local results = inputs
    for i = 1, #net do
        results = iterateLayer(results,net[i])
    end
    return results
end

--Randomizes the input net 'net' until delta threshhold 'threshhold' is reached or iteration maximum 'iterations' is reached.
--Returns: results,prevDelta,iterCount,true/false
--Where results a 1d table of results from neural network,
--prevDelta is the ending delta that the training session either ended with or tripped the threshhold,
--iterCount is the number of iterations taken,
--true/false is equal to true if the training ended because the threshhold was reached. False if because iteration max was reached.
local function trainNet(iterations,threshhold,inputs,expected,net,weightStep,biasStep)
    local prevNet = net
    local prevResults = iterateNet( inputs,prevNet )
    local prevDelta = getResultDelta( prevResults, expected )
    local iterCount = 0

    for i = 1, iterations do

        if prevDelta*1000 < threshhold*1000 then
            --Net is accurate enough to stop training
                --last value is true if returns because threshhold was reached
            return prevResults,prevDelta,prevNet,iterCount,true
        end
        local scrambledNet = netScramble(prevNet,weightStep,biasStep)
        local results = iterateNet( inputs,scrambledNet )
        local delta = getResultDelta( results, expected )
        if delta*1000 < prevDelta*1000 then
            --Scrambeled net is better than old net, so replace it
            prevNet = scrambledNet
            prevResults = results
            prevDelta = delta
        end
        iterCount = i

        term.clear()
        term.setCursorPos(1,1)
        print("Best Delta so far: "..prevDelta)
        print("Current Delta: "..delta)
        print("Results:")
        print(textutils.serialise(results))
        print(textutils.serialise(scrambledNet))
        sleep(0.05)

    end
    --last value is false if returns because iteration max was reached
    return prevResults,prevDelta,prevNet,iterCount,false
end

return {
    makeLayer = makeLayer,
    makeNeuralNet = makeNeuralNet,
    sigmoid = sigmoid,
    heavySide = heavySide,
    activationFunction = activationFunction,
    getResultDelta = getResultDelta,
    netScramble = netScramble,
    iterateLayer = iterateLayer,
    iterateNet = iterateNet,
    trainNet = trainNet
}
