--Example 1
local inputs = {
    { 1,1 }, 
    { 1,0 },
    { 0,1 },
    { 0,0 }
}
local expected = {
    { 0,1,1,1 },
    { 1,0,1,1 },
    { 1,1,0,1 },
    { 1,1,1,0 },
}
local net = makeNeuralNet(2,4,4)
local results,delta,newNet,iterCount,isThreshhold = trainNet(1,0.02,inputs[1],expected[1],net,0.25,0.25)
for j = 1, 2000 do
    for i = 1, #inputs do
        results,delta,newNet,iterCount,isThreshhold = trainNet(6,0.02,inputs[i],expected[i],newNet,0.25,0.25)
    end
end
if isThreshhold then
    isThreshhold = "true"
else
    isThreshhold = "false"
end

local function cprint(c,t)
    local col = term.getTextColor()
    term.setTextColor(c)
    print(t)
    term.setTextColor(col)
end

term.clear()
term.setCursorPos(1,1)

cprint(colors.red,"Finished training!")
cprint(colors.orange,"Ending Iteration: "..iterCount)
cprint(colors.yellow,"Ending Delta: "..delta)
cprint(colors.red,"Ended because threshhold was reached?: ")
cprint(colors.orange,isThreshhold)

cprint(colors.green, "What would you like to do now? (Enter number of command)")
print("1. Test Net")
local doNow = read()
if doNow == "1" then
    while true do
        cprint(colors.red,"Which input would you like to test the net on?")
        for i = 1, #inputs do
            cprint(colors.purple,i..":")
            cprint(colors.blue,textutils.serialise(inputs[i]))
        end
        cprint(colors.red,(#inputs+1)..": Exit")

        doNow = read()
        term.clear()
        term.setCursorPos(1,1)
        if tonumber(doNow) == (#inputs+1) then
            print("Exiting...")
            return
        else
            print("Inputting: ")
            print(textutils.serialise(inputs[tonumber(doNow)]))
            local testResults = iterateNet(inputs[tonumber(doNow)],newNet)
            cprint(colors.orange,"Test Results:")
            print(textutils.serialize(testResults))
        end
    end
end
