--Example file. Requires main function library to run
local lmll = require("perceptron")

local omnius = lmll.makeNet(2,5,6,2)
--lmll.netDebug(omnius)

local inputs = { 
    {10,5},
    {20,10},
    {2,30},
    {10,20},
    {8,9},

    {30,10},
    {40,20},
    {28,12},
    {7,31},
    {15,26},
}

local expectedOutputs = {
    {1,0},
    {1,0},
    {1,0},
    {1,0},
    {1,0},

    {0,1},
    {0,1},
    {0,1},
    {0,1},
    {0,1},
}

local targetCost = 0.05
local cost = lmll.getAvgNetCost(inputs,expectedOutputs,omnius)
--hold Ctr + t to stop the loop at any point!
while cost > targetCost do
    lmll.trainNet(inputs,expectedOutputs,omnius,0.01)
    cost = lmll.getAvgNetCost(inputs,expectedOutputs,omnius)
    print("New Cost: "..cost)
    sleep()
end
print("Finished training!")
print("Resulting net has an cost of "..cost)
lmll.netDebug(omnius)
