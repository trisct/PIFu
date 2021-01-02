# Modification notes

### Use pyembree
pyembree in used in both PRT and training-stage `mesh.contains`. Remember to use it, otherwise it becomes extremely slow.

### Train with this
```
python -m apps.train_shape_hg --dataroot training_data/ --random_flip --random_scale --random_trans --checkpoints_path bike --results_path bike --pitches 0,10,20,30,40,50,60
```
