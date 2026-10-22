import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

sys.path.insert(0, "/root/ana")
import mujoco
from agi.body_go2 import Go2Body
from agi.dog_tasks import _ball, _brain, _obs

kw = json.load(open("/root/ana/walk_seed3.json", encoding="utf-8"))["kw"]
sp, brain = _brain(kw)
eng = brain.engine
body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
body.reset()
body.data.qpos[2] = 0.24
body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
body.mj.mj_forward(body.model, body.data)

renderer = mujoco.Renderer(body.model, height=240, width=320)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance = 2.2
cam.azimuth = 120
cam.elevation = -20
cam.lookat[:] = body.data.qpos[0:3]

os.makedirs("/root/ana/frames", exist_ok=True)
for t in range(120):
    eng.clear_external()
    eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
    eng.step()
    body.step(brain.forces())
    if t % 30 == 0:
        cam.lookat[:] = body.data.qpos[0:3]
        renderer.update_scene(body.data, cam)
        img = renderer.render()
        import imageio.v2 as imageio
        imageio.imwrite("/root/ana/frames/walk_%02d.png" % t, img)
print("done", sorted(os.listdir("/root/ana/frames"))[:4])
