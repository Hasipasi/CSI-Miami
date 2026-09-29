"""Shared pieces of the 3-D pose pipeline: the SMPL body model as a differentiable
torch layer, the RealSense camera geometry (colour intrinsics, depth intrinsics,
depth-to-colour extrinsics), and the depth-frame-to-point-cloud step.

Used by extract_poses3d.py (RGB -> per-frame SMPL initialisation), fit_body.py
(depth-grounded SMPL fit + temporal bundle adjustment) and plot_body_fit.py.

Coordinate conventions, everywhere in this pipeline:
  * metres, in the COLOUR camera's optical frame: x right, y down, z forward, as
    librealsense defines it. The camera does not move during a session, so this is
    a fixed room frame up to one rigid transform.
  * pixels index the 1280x720 colour image; (u, v) = (fx X/Z + ppx, fy Y/Z + ppy).
  * SMPL pose is 24 axis-angle rotations flattened to 72, root first; betas are 10
    shape coefficients; trans is added to the posed vertices and joints.

The SMPL model data (template, blend shapes, regressor, skinning weights) is not
redistributable, so it is not in the repo. `smpl_layer()` loads it from
models/smpl_neutral.npz, which extract_poses3d.py writes once from the body model
embedded in the NLF torchscript, or from the official SMPL pkl via the smplfitter
package if that is installed and the files are in body_models/smpl/.
"""

import json
import os
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / 'models'
SMPL_NPZ = MODELS_DIR / 'smpl_neutral.npz'

COCO17 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
COCO_SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
                 (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (0, 1), (0, 2), (1, 3), (2, 4)]

SMPL24 = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
          'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
          'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
          'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
          'left_hand', 'right_hand']
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18,
                19, 20, 21]

# COCO-17 keypoints out of an SMPL body. The limb joints are SMPL joints (the SMPL
# hip is inside the pelvis, a few cm above and medial to where a 2-D detector puts
# the hip -- a consistent offset, not noise). The five face points have no SMPL
# joint and come from mesh vertices, the ids smplx uses for the same purpose.
COCO_FROM_SMPL = {0: ('v', 332), 1: ('v', 2800), 2: ('v', 6260), 3: ('v', 583),
                  4: ('v', 4071), 5: ('j', 16), 6: ('j', 17), 7: ('j', 18), 8: ('j', 19),
                  9: ('j', 20), 10: ('j', 21), 11: ('j', 1), 12: ('j', 2), 13: ('j', 4),
                  14: ('j', 5), 15: ('j', 7), 16: ('j', 8)}
# Torso joints used to read the person's distance off the depth frame before the
# fit starts: the body parts a depth camera sees best and that move least.
TORSO_JOINTS = [0, 3, 6, 9, 12, 16, 17, 1, 2]

# -------------------------------------------------------------------- room frame

# The rig, in metres. Origin on the floor at the centre of the board square, X to
# the camera's right, Y up, Z away from the camera. The camera sits on board A's
# rod and looks along the A->C diagonal, so the centre of a 3 m square is half a
# diagonal away -- 2.12 m, which is the ~2.1 m paced out on the floor. Antennas
# ride the rods at 1.20 m, the camera 20 cm below the one it shares with A.
ROOM_SIDE = 3.0
HALF_DIAG = ROOM_SIDE * np.sqrt(2) / 2          # 2.121 m: camera to arena centre
BOARD_H, CAM_H = 1.20, 1.00
# (x, z) on the floor. Seen from the camera at A: C straight ahead, B left, D right
# (README setup 3 -- far row C D, near row B A).
BOARDS = {'A': (0.0, -HALF_DIAG), 'B': (-HALF_DIAG, 0.0),
          'C': (0.0, HALF_DIAG), 'D': (HALF_DIAG, 0.0)}
CAMERA_XZ = BOARDS['A']


def room_from_camera(pitch_deg=0.0, cam_height=CAM_H, centre_dist=HALF_DIAG):
    """(R, t) taking colour-camera points (x right, y down, z forward) to the room
    frame: X = R @ P + t.

    Nothing measures the camera's tilt, so pitch is a knob rather than a constant:
    a couple of degrees of downward tilt moves the far corner of the arena by ~10 cm
    and is the first thing to try when the fitted feet do not sit near Y = 0.
    """
    c, s = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    # camera x -> X; camera "down" -> (0, -c, -s); camera "forward" -> (0, -s, c)
    R = np.array([[1.0, 0.0, 0.0], [0.0, -c, -s], [0.0, -s, c]])
    return R, np.array([0.0, cam_height, -centre_dist])


def to_room(P, pitch_deg=0.0, cam_height=CAM_H, centre_dist=HALF_DIAG):
    """Camera-frame points [..., 3] -> room frame, NaNs surviving as NaNs."""
    R, t = room_from_camera(pitch_deg, cam_height, centre_dist)
    return np.asarray(P, float) @ R.T + t


def _demo():
    a = to_room(np.array([[0.0, 0.0, 0.0], [0.0, CAM_H, HALF_DIAG]]))
    assert np.allclose(a[0], [0, CAM_H, -HALF_DIAG]), a[0]      # the camera itself
    assert np.allclose(a[1], [0, 0, 0]), a[1]                   # floor, arena centre
    # a downward tilt puts a point on the optical axis lower and nearer, and a
    # 3 m square really does have a 2.12 m half-diagonal
    p = to_room(np.array([0.0, 0.0, 2.0]), pitch_deg=10.0)
    assert p[1] < CAM_H and p[2] < 2.0 - HALF_DIAG, p
    assert abs(np.hypot(*BOARDS['A']) - HALF_DIAG) < 1e-9
    assert abs(np.hypot(*(np.array(BOARDS['A']) - BOARDS['B'])) - ROOM_SIDE) < 1e-9
    print('room frame OK')


# ------------------------------------------------------------------ calibration

# Nominal D435i geometry for takes recorded before the calibration was stored in
# meta (2026-09-17). Colour 1280x720 and depth 640x480 focal lengths are the
# module's typical values, and the depth->colour offset is the usual factory
# value. Wrong by up to a percent or so -- run tools/rs_calib.py on the rig once
# and pass --calib to replace it; the tools print loudly when they fall back.
D435I_NOMINAL = dict(
    serial=None, source='nominal D435i values, NOT measured',
    colour_intrinsics=dict(width=1280, height=720, fx=915.0, fy=915.0,
                           ppx=640.0, ppy=360.0, model='nominal', coeffs=[0.0] * 5),
    depth_intrinsics=dict(width=640, height=480, fx=385.0, fy=385.0,
                          ppx=320.0, ppy=240.0, model='nominal', coeffs=[0.0] * 5),
    depth_to_colour=dict(rotation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                         translation=[0.0148, 0.0, 0.0]),
    depth_scale_m=0.001)


def default_calib_path(serial=None):
    d = REPO / 'calib'
    if serial:
        p = d / f'realsense_{serial}.json'
        if p.exists():
            return p
    found = sorted(d.glob('realsense_*.json')) if d.is_dir() else []
    return found[0] if found else None


def load_calib(meta=None, calib_path=None, verbose=True):
    """The camera geometry for a take: what its meta recorded, filled in from a
    calib JSON (tools/rs_calib.py) or, last, the nominal table above.

    Returns a dict with colour_intrinsics, depth_intrinsics, depth_to_colour
    (rotation row-major 9, translation 3, metres, depth frame -> colour frame),
    depth_scale_m, and `source` saying where each part came from.
    """
    meta = meta or {}
    out = json.loads(json.dumps(D435I_NOMINAL))
    src = {k: 'nominal' for k in ('colour_intrinsics', 'depth_intrinsics', 'depth_to_colour')}
    path = pathlib.Path(calib_path) if calib_path else default_calib_path(meta.get('camera_serial'))
    if path is not None and path.exists():
        with open(path) as fh:
            c = json.load(fh)
        for k in src:
            if k in c:
                out[k] = c[k]
                src[k] = f'calib:{path.name}'
        if 'depth_scale_m' in c:
            out['depth_scale_m'] = c['depth_scale_m']
        out['serial'] = c.get('serial')
    # What the recorder wrote wins: it came from the very device and stream mode.
    if meta.get('depth_intrinsics'):
        di = dict(out['depth_intrinsics'])
        di.update(meta['depth_intrinsics'])
        di['width'] = meta.get('depth_width', di.get('width'))
        di['height'] = meta.get('depth_height', di.get('height'))
        out['depth_intrinsics'], src['depth_intrinsics'] = di, 'meta'
    if meta.get('colour_intrinsics'):
        out['colour_intrinsics'], src['colour_intrinsics'] = meta['colour_intrinsics'], 'meta'
    if meta.get('depth_to_colour'):
        out['depth_to_colour'], src['depth_to_colour'] = meta['depth_to_colour'], 'meta'
    if meta.get('depth_scale_m'):
        out['depth_scale_m'] = float(meta['depth_scale_m'])
    out['source'] = src
    if verbose and any(v == 'nominal' for v in src.values()):
        nominal = [k for k, v in src.items() if v == 'nominal']
        print(f'  WARNING: {", ".join(nominal)} not in meta and no calib JSON found: '
              f'using nominal D435i values. Run tools/rs_calib.py on the rig and pass '
              f'--calib for metric accuracy.', flush=True)
    return out


def K_of(intr):
    """3x3 pinhole matrix from a librealsense-style intrinsics dict."""
    return np.array([[intr['fx'], 0.0, intr['ppx']],
                     [0.0, intr['fy'], intr['ppy']],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def project(K, X):
    """[..., 3] camera-frame points -> [..., 2] pixels. Points behind the camera
    project to NaN rather than to a mirrored position."""
    X = np.asarray(X, dtype=np.float64)
    z = X[..., 2:3]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = X[..., :2] / np.where(z > 1e-6, z, np.nan)
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


# ------------------------------------------------------------- depth -> points

def depth_to_points(depth_u16, calib, z_min=0.25, z_max=6.0):
    """A depth frame (uint16 device units) as points in the COLOUR camera frame.

    Returns (points [N, 3] float32 metres, depth pixel index [N] int) for every
    valid depth pixel within [z_min, z_max]. The depth stream is rectified so its
    distortion is ignored; the colour stream's is tiny on a D435i and ignored too.
    """
    di = calib['depth_intrinsics']
    d = np.asarray(depth_u16)
    h, w = d.shape
    z = d.astype(np.float32) * float(calib['depth_scale_m'])
    ok = (z >= z_min) & (z <= z_max)
    idx = np.flatnonzero(ok)
    v, u = np.divmod(idx, w)
    zz = z.reshape(-1)[idx]
    x = (u - di['ppx']) / di['fx'] * zz
    y = (v - di['ppy']) / di['fy'] * zz
    P = np.stack([x, y, zz], axis=1)
    ex = calib['depth_to_colour']
    R = np.asarray(ex['rotation'], dtype=np.float32).reshape(3, 3)
    t = np.asarray(ex['translation'], dtype=np.float32)
    # librealsense stores the rotation column-major: X_colour = R^T-as-stored @ X_depth
    # in row-major terms, i.e. exactly the transform rs2_transform_point_to_point
    # applies. Written out so a reader can check it against that function.
    Pc = P @ R + t              # (R.T @ p) for row-major R == p @ R
    return Pc.astype(np.float32), idx


def read_depth_png(zf, member):
    """One 16-bit depth PNG out of a capture archive, as uint16 [H, W]."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(zf.read(member)))
    a = np.asarray(im)
    if a.dtype != np.uint16:
        a = a.astype(np.uint16)
    return a


def read_jpeg(zf, member):
    import io
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(zf.read(member))).convert('RGB'))


def frame_members(d, meta):
    """(colour member or None, depth member or None) per kept frame of a capture,
    following the pairing the writer stored (`frame_depth_idx`)."""
    fidx = d['frame_idx'].astype(np.int64)
    gt = meta.get('gt', 'colour')
    ext = '.png' if gt == 'depth' else '.jpg'
    colour = [None if gt == 'depth' else f'frames/{int(i):06d}.jpg' for i in fidx]
    if gt == 'depth':
        depth = [f'frames/{int(i):06d}.png' for i in fidx]
    elif 'frame_depth_idx' in d.files:
        depth = [f'depth/{int(i):06d}.png' if i >= 0 else None
                 for i in d['frame_depth_idx'].astype(np.int64)]
    else:
        depth = [None] * len(fidx)
    return colour, depth, ext


# ---------------------------------------------------------------- SMPL layer

def rodrigues(rv):
    """Axis-angle [..., 3] -> rotation matrices [..., 3, 3] (torch)."""
    import torch
    angle = torch.linalg.norm(rv, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = rv / angle
    c = torch.cos(angle)[..., None]
    s = torch.sin(angle)[..., None]
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    zero = torch.zeros_like(x)
    Kx = torch.stack([torch.cat([zero, -z, y], -1), torch.cat([z, zero, -x], -1),
                      torch.cat([-y, x, zero], -1)], -2)
    I = torch.eye(3, dtype=rv.dtype, device=rv.device).expand(Kx.shape)
    return I + s * Kx + (1 - c) * (Kx @ Kx)


class SMPL:
    """Linear blend skinning SMPL with 24 joints, written out so the fit has no
    dependency beyond torch and one npz of model arrays. Verified against the
    vertices NLF returns for the same parameters (extract_poses3d.py --check-smpl).

    forward(pose [F, 72], betas [F, 10] or [10], trans [F, 3]) ->
        vertices [F, 6890, 3], joints [F, 24, 3]   (metres, camera frame)
    """

    def __init__(self, path=SMPL_NPZ, device='cpu', dtype=None):
        import torch
        self.torch = torch
        dtype = dtype or torch.float32
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f'{path} missing. Run extract_poses3d.py once (it writes the SMPL arrays '
                f'out of the NLF model), or export_smpl_npz() from an smplfitter BodyModel.')
        z = np.load(path)
        t = lambda k: torch.as_tensor(np.asarray(z[k]), dtype=dtype, device=device)
        self.v_template = t('v_template')                    # [V, 3]
        self.shapedirs = t('shapedirs')                      # [V, 3, S]
        self.posedirs = t('posedirs')                        # [V, 3, 207]
        # Joint locations of the shaped rest body, either regressed from the
        # vertices (the original pkl) or as a template plus shape directions (the
        # precomputed form NLF/smplfitter carry). The two are the same linear map.
        if 'J_template' in z.files:
            self.J_template, self.J_shapedirs = t('J_template'), t('J_shapedirs')
            self.J_regressor = None
        else:
            self.J_regressor = t('J_regressor')              # [24, V]
            self.J_template = self.J_regressor @ self.v_template
            self.J_shapedirs = torch.einsum('jv,vcs->jcs', self.J_regressor, self.shapedirs)
        self.weights = t('weights')                          # [V, 24]
        self.parents = [int(p) for p in np.asarray(z['kintree_parents'])]
        # Triangles are only needed for drawing; the fit is point-to-vertex.
        self.faces = (np.asarray(z['faces']).astype(np.int64)
                      if 'faces' in z.files and np.asarray(z['faces']).size else None)
        self.num_betas = int(self.shapedirs.shape[2])
        self.V = int(self.v_template.shape[0])
        self.device, self.dtype = device, dtype

    def forward(self, pose, betas, trans, return_vertices=True):
        torch = self.torch
        F = pose.shape[0]
        if betas.dim() == 1:
            betas = betas[None].expand(F, -1)
        nb = min(betas.shape[1], self.num_betas)
        v_shaped = self.v_template[None] + torch.einsum(
            'vcs,fs->fvc', self.shapedirs[:, :, :nb], betas[:, :nb])
        J = self.J_template[None] + torch.einsum(
            'jcs,fs->fjc', self.J_shapedirs[:, :, :nb], betas[:, :nb])      # [F, 24, 3]
        R = rodrigues(pose.reshape(F, 24, 3))                              # [F, 24, 3, 3]
        I = torch.eye(3, dtype=pose.dtype, device=pose.device)
        # forward kinematics: world transform of every joint
        G = [None] * 24
        G[0] = self._rt(R[:, 0], J[:, 0])
        for k in range(1, 24):
            p = self.parents[k]
            G[k] = G[p] @ self._rt(R[:, k], J[:, k] - J[:, p])
        G = torch.stack(G, dim=1)                                          # [F, 24, 4, 4]
        joints = G[:, :, :3, 3] + trans[:, None]
        if not return_vertices:
            return None, joints
        pose_feature = (R[:, 1:] - I).reshape(F, 207)
        v_posed = v_shaped + torch.einsum('vcp,fp->fvc', self.posedirs, pose_feature)
        # skinning transforms relative to the rest pose
        Jh = torch.cat([J, torch.zeros_like(J[:, :, :1])], dim=-1)         # [F, 24, 4]
        Gr = G.clone()
        Gr[:, :, :3, 3] = G[:, :, :3, 3] - torch.einsum('fkab,fkb->fka', G[:, :, :3, :3], J)
        T = torch.einsum('vk,fkab->fvab', self.weights, Gr)                # [F, V, 4, 4]
        verts = torch.einsum('fvab,fvb->fva', T[:, :, :3, :3], v_posed) + T[:, :, :3, 3]
        return verts + trans[:, None], joints

    def _rt(self, R, t):
        torch = self.torch
        F = R.shape[0]
        M = torch.zeros(F, 4, 4, dtype=R.dtype, device=R.device)
        M[:, :3, :3] = R
        M[:, :3, 3] = t
        M[:, 3, 3] = 1.0
        return M

    def coco17(self, verts, joints):
        """COCO-17 keypoints [F, 17, 3] from a forward() result."""
        torch = self.torch
        out = []
        for i in range(17):
            kind, idx = COCO_FROM_SMPL[i]
            out.append(verts[:, idx] if kind == 'v' else joints[:, idx])
        return torch.stack(out, dim=1)


def export_smpl_npz(arrays, path=SMPL_NPZ, source='unknown', pose_feature='R-I'):
    """Write the SMPL arrays in the layout SMPL() reads, normalising the shapes the
    various sources use (smplfitter keeps posedirs as [V, 3, 207]; the original pkl
    as [V, 3, 207] too but shapedirs as chumpy; some exports flatten).

    `pose_feature` names the convention of the SOURCE: the original SMPL drives the
    pose blend shapes with vec(R - I) per joint, while smplfitter (and so the body
    model inside NLF) uses vec(R) with a template that has the identity's
    contribution subtracted out. The two agree once that contribution is put back
    into the template, which is done here so SMPL() can use the textbook form.
    """
    a = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
    V = a['v_template'].shape[0]
    if pose_feature == 'R':
        pd = a['posedirs']
        if pd.shape[0] != V:
            pd = np.transpose(pd, (1, 2, 0))
        eye = np.tile(np.eye(3, dtype=np.float64).reshape(-1), 23)          # [207]
        a['v_template'] = a['v_template'] + np.einsum('vcp,p->vc', pd, eye)
        a['posedirs'] = pd
    elif pose_feature != 'R-I':
        raise ValueError(pose_feature)
    sd = a['shapedirs']
    if sd.shape[0] != V:                     # [S, V, 3] -> [V, 3, S]
        sd = np.transpose(sd, (1, 2, 0))
    pd = a['posedirs']
    if pd.ndim == 2:                         # [207, V*3] -> [V, 3, 207]
        pd = pd.reshape(pd.shape[0], V, 3).transpose(1, 2, 0)
    elif pd.shape[0] != V:
        pd = np.transpose(pd, (1, 2, 0))
    joints = {}
    if 'J_template' in a and 'J_shapedirs' in a:
        js = a['J_shapedirs']
        if js.shape[0] != 24:                # [S, 24, 3] -> [24, 3, S]
            js = np.transpose(js, (1, 2, 0))
        joints = dict(J_template=a['J_template'].astype(np.float32),
                      J_shapedirs=js.astype(np.float32))
    else:
        jr = a['J_regressor']
        if jr.shape[0] == V:
            jr = jr.T
        joints = dict(J_regressor=jr.astype(np.float32))
    w = a['weights']
    if w.shape[0] != V:
        w = w.T
    parents = np.asarray(a['kintree_parents']).reshape(-1)
    if parents.shape[0] != 24:               # kintree_table [2, 24]
        parents = np.asarray(a['kintree_parents'])[0]
    parents = parents.astype(np.int64)
    parents[0] = -1                          # stored as uint32 0xffffffff by some
    if list(parents) != SMPL_PARENTS:
        raise ValueError(f'unexpected SMPL kinematic tree: {parents.tolist()}')
    faces = a.get('faces')
    faces = (np.asarray(faces).astype(np.int64) if faces is not None
             else np.zeros((0, 3), np.int64))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, v_template=a['v_template'].astype(np.float32),
                        shapedirs=sd.astype(np.float32), posedirs=pd.astype(np.float32),
                        weights=w.astype(np.float32), kintree_parents=parents,
                        faces=faces, source=np.array(source), **joints)
    return path


def smpl_from_nlf(model):
    """The SMPL arrays embedded in an NLF multi-person torchscript (its
    smplfitter BodyModel submodule), as a dict of numpy arrays. Raises if the
    scripted module does not expose them under the expected names."""
    bm = None
    try:
        bm = model.body_models.smpl
    except AttributeError:
        for name, mod in model.named_modules():
            if name.endswith('body_models.smpl'):
                bm = mod
                break
    if bm is None:
        raise RuntimeError('no body_models.smpl submodule in this NLF model')
    bufs = {k: v.detach().cpu().numpy() for k, v in bm.named_buffers()}
    need = ['v_template', 'shapedirs', 'posedirs', 'weights', 'J_template', 'J_shapedirs']
    missing = [k for k in need if k not in bufs]
    if missing:
        raise RuntimeError(f'NLF body model lacks {missing}; has {sorted(bufs)}')
    parents = bufs.get('kintree_parents_tensor', bufs.get('kintree_parents'))
    if parents is None:
        parents = np.array(SMPL_PARENTS)
    parents = np.asarray(parents).astype(np.int64)
    parents[0] = -1
    # The scripted module carries no triangle list; drawing falls back to points.
    return dict(v_template=bufs['v_template'], shapedirs=bufs['shapedirs'],
                posedirs=bufs['posedirs'], J_template=bufs['J_template'],
                J_shapedirs=bufs['J_shapedirs'], weights=bufs['weights'],
                kintree_parents=parents, faces=None)


def smpl_from_smplfitter():
    """The same arrays via `pip install smplfitter` and the official SMPL files in
    body_models/smpl/ (see the smplfitter README for the lookup path)."""
    from smplfitter.pt import BodyModel
    bm = BodyModel('smpl', 'neutral', num_betas=10)
    bufs = {k: v.detach().cpu().numpy() for k, v in bm.named_buffers()}
    out = dict(v_template=bufs['v_template'], shapedirs=bufs['shapedirs'],
               posedirs=bufs['posedirs'], weights=bufs['weights'],
               kintree_parents=np.asarray(list(bm.kintree_parents)),
               faces=np.asarray(getattr(bm, 'faces', None)))
    for k in ('J_regressor', 'J_template', 'J_shapedirs'):
        if k in bufs:
            out[k] = bufs[k]
    return out


def lerp_rotvec(a, b, w):
    """Interpolate axis-angle vectors [..., 3] by w in [0, 1] through the shortest
    rotation, per joint. Used only to fill undetected frames before the fit."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    shp = a.shape
    a2, b2 = a.reshape(-1, 3), b.reshape(-1, 3)
    out = np.empty_like(a2)
    for i in range(len(a2)):
        s = Slerp([0.0, 1.0], Rot.from_rotvec(np.stack([a2[i], b2[i]])))
        out[i] = s([w]).as_rotvec()[0]
    return out.reshape(shp)


if __name__ == '__main__':
    _demo()
