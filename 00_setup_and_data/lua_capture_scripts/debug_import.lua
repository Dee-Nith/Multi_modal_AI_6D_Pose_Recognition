-- Debug Import Script - Test file access
print("🔍 Debugging YCB model import...")

-- Test if we can access the directory
local testPath = "/Users/nith/Desktop/AI_6D_Pose_recognition/Data_sets/YCB-Video-Base/models/002_master_chef_can/"
print("Testing directory access: " .. testPath)

-- Try to import with more detailed error checking
local objPath = testPath .. "textured.obj"
local mtlPath = testPath .. "textured.mtl"

print("OBJ file path: " .. objPath)
print("MTL file path: " .. mtlPath)

-- Try importing with different options
local importOptions = {
    {0, objPath, 0, 0, 0},           -- Standard import
    {0, objPath, 0, 0, 1},           -- With scaling
    {0, objPath, 0, 0, 2},           -- With different options
}

for i, options in ipairs(importOptions) do
    print("Attempt " .. i .. " with options: " .. table.concat(options, ", "))
    local result = sim.importShape(table.unpack(options))
    print("Import result: " .. tostring(result))
    
    if result ~= -1 then
        print("✅ SUCCESS! Model imported with handle: " .. result)
        sim.setObjectName(result, 'Master Chef Can - Attempt ' .. i)
        break
    else
        print("❌ Failed attempt " .. i)
    end
end

-- Try a simple shape creation to test if importShape works at all
print("Testing basic shape creation...")
local testShape = sim.createPureShape(0, 0, {0.1, 0.1, 0.1}, 0, nil)
if testShape ~= -1 then
    print("✅ Basic shape creation works: " .. testShape)
    sim.removeObject(testShape)
else
    print("❌ Basic shape creation failed")
end

print("🎉 Debug completed!")







