# Code Notes of apps.render_data

### 1. trimesh load problem (Updated: 2020/12/05)

Trimesh loads an OBJ file using `trimesh.load`. Sometimes the returned object is a real mesh object, but sometimes it is a scene object. The issue is mentioned in https://github.com/mikedh/trimesh/issues/507. I used the suggestion by jackd (2019/07/25) in `apps/prt_util.py`.