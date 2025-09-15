-- Test Import with Copied Files
print("🧪 Testing import with copied files...")

-- Use the copied files on Desktop
local objPath = "/Users/nith/Desktop/ycb_test/textured.obj"
print("Trying to import: " .. objPath)

local result = sim.importShape(0, objPath, 0, 0, 0)
print("Import result: " .. tostring(result))

if result ~= -1 then
    print("✅ SUCCESS! Model imported!")
    sim.setObjectName(result, 'Master Chef Can')
    print("Model handle: " .. result)
else
    print("❌ Import failed")
    
    -- Try alternative approach
    print("Trying alternative import method...")
    local altResult = sim.importShape(0, objPath, 0, 0, 1)
    print("Alternative result: " .. tostring(altResult))
end

print("�� Test completed!")







