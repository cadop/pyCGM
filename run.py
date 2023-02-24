import os

import pycgm

script_dir = pycgm.get_data_dir()

# Standard Model, 2 dynamic trials
# model = Model(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
#              [os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), os.path.join(script_dir, 'ROM/Sample_Dynamic.c3d')], \
#               os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))

model = pycgm.Model(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
                    os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), \
                    os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))

# Run model
model.run()

# Verify RoboWalk results
diff_pycgm_csv(model, 'RoboWalk', os.path.join(script_dir, 'Sample_2/pycgm_results.csv'))

# Run set of models
# cgm = pycgm.ModelSet([model, model, model, model])
# cgm.run_all()

# Access model output
print(f"{model.data.dynamic.RoboWalk.axes.Pelvis.shape=}")
