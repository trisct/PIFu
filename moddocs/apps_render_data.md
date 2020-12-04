# Code Notes of apps.render_data

Note that the line numbers marked here may vary due to modifications of the original code.

### 1. Prt usage

This app uses the prt computed by `apps.prt_util` saved in `bounce`. This can cause a problem if a mesh is low-poly. Turn it off by setting the loaded prt array to all ones. Approximately at lines 205-212. I added an option `--use_prt`.

Modified part:
```
face_prt_file = os.path.join(folder_name, 'bounce', 'face.npy')

if args.use_prt:
    if not os.path.exists(face_prt_file):
        print('ERROR: face prt file does not exist!!!', prt_file)
        return
    else:
        print('HERE: you have specified usr_prt.')
        print('HERE: will use face_prt file %s' % face_prt_file)
```

```
###### prt loading
# this has a potential problem if the mesh is low-poly
# a large area would have the same texture and the output would appear broken
# using a all-one tensor instead gets rid of the prt effects
prt = np.loadtxt(prt_file)
face_prt = np.load(face_prt_file)
print('face_prt shape = ', face_prt.shape)
if not args.use_prt:    
    face_prt = np.ones_like(face_prt)
###### thinking of adding an arg to make this optional
```

### 2. Object orientation, coordinate normalization and rotations during rendering

Object orientation in rendering is set by `up_axis`, which defines what axis is used to point 'up' during rendering. RenderPeople dataset usually have the y-axis pointint up. But bike models usually have the z-axis pointing up. So I changed up_axis manually to z-axis-up always.

Coodinate normalization is set is the renderer as a normalization matrix, by calling `rndr.set_norm_mat` and `rndr_uv.set_norm_mat`. The scaling it was originally set along the y-axis, i.e. the longest axis of a person. But for bike models, the longest axis is not facing up! So I added a `longest_axis` instead of using `up_axis`. The scaling target is 180, because the near-far range is by default -100~100, so 180 takes up 90% of the range.

Somehow the orientation is wrong even after changing `up_axis` to 2. This is caused by using a wrong rotation matrix (perhaps in RenderPeople, if the longest axis is along z, it would be correct). I added an addition rotation along z-axis by 180 degrees to solve this problem.

### 3. Cutoff range

This app has a min-max range of visible vertices along the direction of the sight. This is approximately at lines 151-155. One might need to set a larger range for some long bikes. The default range is `near = -100` and `far = 100`. 

Modified part:
```
###### range of the mesh visible is specified here
# default: -100 ~ 100
cam.near = -100
cam.far = 100
######
```