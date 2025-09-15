-- Simple YCB Model Import
-- Copy this entire script into CoppeliaSim

function importModels()
    print("🤖 Starting YCB model import...")
    
    -- Create container
    local container = sim.createPureShape(0, 0, {0.1, 0.1, 0.1}, 0, nil)
    sim.setObjectName(container, 'YCB_Models')
    
    -- Import Master Chef Can
    local can = sim.importShape(0, '../../Data_sets/YCB-Video-Base/models/002_master_chef_can/textured.obj', 0, 0, 0)
    if can ~= -1 then
        sim.setObjectName(can, 'Master Chef Can')
        sim.setObjectParent(can, container, true)
        print('✅ Imported: Master Chef Can')
    end
    
    -- Import Cracker Box
    local box = sim.importShape(0, '../../Data_sets/YCB-Video-Base/models/003_cracker_box/textured.obj', 0, 0, 0)
    if box ~= -1 then
        sim.setObjectName(box, 'Cracker Box')
        sim.setObjectParent(box, container, true)
        print('✅ Imported: Cracker Box')
    end
    
    print('🎉 Import completed!')
end

importModels()







