# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

import json
import os

import numpy as np


def prepare_sample_scene(scene, path, path_meta, verbose=2):
    """Generates a json file for a sample scene in our common format

    This wraps all the data about a scene into a single common format
    that is readable by our dataset class. This includes paths to all
    the images, depths, etc, as well as other metadata like camera
    intrinsics and pose. It also includes scene level information like
    a path to the mesh.

    Args:
        scene: name of the scene.
            examples: 'scans/scene0000_00'
                      'scans_test/scene0708_00'
        path: path to the original data
        path_meta: path to where the generated data is saved.
            This can be the same as path, but it is recommended to
            keep them seperate so the original data is not accidentally
            modified. The generated data is saved into a mirror directory
            structure.

    Output:
        Creates the file path_meta/scene/info.json
        
    JSON format:
        {'dataset': 'sample',
         'path': path,
         'scene': scene,
         'frames': [{'file_name_image': '',
                     'intrinsics': intrinsics,
                     'pose': pose,
                     }
                   ]
         }

    """

    if verbose>0:
        print('preparing %s'%scene)

    data = {'dataset': 'sample',
            'path': path,
            'scene': scene,
            'frames': []
           }
    
    intrinsics = np.loadtxt(os.path.join(path, scene, 'intrinsics.txt'))

    frame_ids = os.listdir(os.path.join(path, scene, 'color'))
    frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
    frame_ids =  sorted(frame_ids)

    for i, frame_id in enumerate(frame_ids):
        if verbose>1 and i%25==0:
            print('preparing %s frame %d/%d'%(scene, i, len(frame_ids)))

        pose = np.loadtxt(os.path.join(path, scene, 'pose', '%08d.txt' % frame_id))

        # skip frames with no valid pose
        if not np.all(np.isfinite(pose)):
            continue

        frame = {'file_name_image': 
                     os.path.join(path, scene, 'color', '%08d.jpg'%frame_id),
                 'intrinsics': intrinsics.tolist(),
                 'pose': pose.tolist(),
                }
        data['frames'].append(frame)

    os.makedirs(os.path.join(path_meta, scene), exist_ok=True)
    json.dump(data, open(os.path.join(path_meta, scene, 'info.json'), 'w'))


def prepare_test_scene(scene, path, path_meta, verbose=2):
    """Generates a json file for a sample scene in our common format

    This wraps all the data about a scene into a single common format
    that is readable by our dataset class. This includes paths to all
    the images, depths, etc, as well as other metadata like camera
    intrinsics and pose. It also includes scene level information like
    a path to the mesh.

    Args:
        scene: name of the scene.
            examples: 'scans/scene0000_00'
                      'scans_test/scene0708_00'
        path: path to the original data
        path_meta: path to where the generated data is saved.
            This can be the same as path, but it is recommended to
            keep them seperate so the original data is not accidentally
            modified. The generated data is saved into a mirror directory
            structure.

    Output:
        Creates the file path_meta/scene/info.json

    JSON format:
        {'dataset': 'sample',
         'path': path,
         'scene': scene,
         'frames': [{'file_name_image': '',
                     'intrinsics': intrinsics,
                     'pose': pose,
                     }
                   ]
         }

    """

    if verbose > 0:
        print('preparing %s' % scene)

    data = {'dataset': 'sample',
            'path': path,
            'scene': scene,
            'frames': []
            }


    with open(os.path.join(path, scene, 'pose.txt')) as f:
        poses = {}
        frame_ids=[]
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            words = line.split(" ")
            intrinsics = np.array(words[1:10],np.float32).reshape(3, 3)
            R = np.array(words[10:19], np.float32).reshape(3, 3)
            t = np.array(words[19:22], np.float32).reshape(3, 1)
            P = np.concatenate((R, t), axis=1)
            P = np.concatenate((P, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
            poses[words[0]] = P
            frame_ids.append(words[0])
            #print(P)

    #frame_ids = os.listdir(os.path.join(path, scene, 'color'))
    #frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
    #frame_ids = sorted(frame_ids)

    for i, frame_id in enumerate(frame_ids):
        if verbose > 1 and i % 25 == 0:
            print('preparing %s frame %d/%d' % (scene, i, len(frame_ids)))

        pose = poses[frame_id]

        # skip frames with no valid pose
        if not np.all(np.isfinite(pose)):
            continue

        frame = {'file_name_image':
                     os.path.join(path, scene, 'color', frame_id),
                 'intrinsics': intrinsics.tolist(),
                 'pose': pose.tolist(),
                 }
        data['frames'].append(frame)

    os.makedirs(os.path.join(path_meta, scene), exist_ok=True)
    json.dump(data, open(os.path.join(path_meta, scene, 'info.json'), 'w'))
