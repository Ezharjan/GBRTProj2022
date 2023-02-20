local function commander(argToExcute)
    local result = io.popen(argToExcute)
    local strInfo = result:read("*all")
    return strInfo
end

function tryTillSucceed(arg,tryTimes)
    tryTimes = tryTimes or 1000
    for i = 1,tryTimes,1 do
        print('argument is: ',arg)
        local res = commander(arg)
        if res ~= '' then
            print('Conduction succeeded!')
            break
        end
    end
    return
end

local commitInfo = '\"auto commit at ostime(' .. string.sub(tostring(os.time()),5) .. ') via gitPusher\"'
local addCmd = 'git add .'
local commitCmd = 'git commit -m ' .. commitInfo
local pushCmd =  'git push -u origin master'
local pullCmd =  'git pull'

tryTillSucceed(pullCmd)
os.execute(addCmd)
os.execute(commitCmd)
tryTillSucceed(pushCmd)
tryTillSucceed(pullCmd)
