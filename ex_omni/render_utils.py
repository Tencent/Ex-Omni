import torch
import numpy as np
import os
import subprocess
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    DirectionalLights,
    TexturesVertex,
    Materials,
)
from pytorch3d.renderer.blending import BlendParams


# Model blendshape 顺序
MODEL_BS_LIST = [
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeWideLeft", "eyeWideRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "browDownLeft", "browDownRight",
    "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "jawOpen",
    "mouthClose", "jawLeft", "jawRight", "jawForward", "mouthUpperUpLeft",
    "mouthUpperUpRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthRollUpper", "mouthRollLower", "mouthSmileLeft", "mouthSmileRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthFrownLeft", "mouthFrownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthFunnel", "mouthLeft", "mouthRight",
    "mouthShrugLower", "mouthShrugUpper", "noseSneerLeft", "noseSneerRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight"
]

Standard_MODEL_BS_LIST = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

def create_blendshape_mapping(TEMPLATE_BS_LIST, model_bs_list):
    """
    创建从模型权重顺序到网格 blendshape 顺序的映射
    Args:
        TEMPLATE_BS_LIST: 网格文件中的 blendshape 名称列表
        model_bs_list: 模型输出的 blendshape 名称列表
    Returns:
        mapping_indices: (52,) 映射索引数组，mapping_indices[i] 表示 model_bs_list[i] 
                        对应 TEMPLATE_BS_LIST 中的索引位置
    """
    # 将 numpy bytes 转换为字符串（如果需要）
    if isinstance(TEMPLATE_BS_LIST[0], bytes):
        TEMPLATE_BS_LIST = [name.decode('utf-8') for name in TEMPLATE_BS_LIST]
    
    # 创建网格名称到索引的映射
    gt_name_to_idx = {name: idx for idx, name in enumerate(TEMPLATE_BS_LIST)}
    
    # 创建映射索引
    mapping_indices = []
    missing_names = []
    
    for model_idx, model_name in enumerate(model_bs_list):
        if model_name in gt_name_to_idx:
            gt_idx = gt_name_to_idx[model_name]
            mapping_indices.append(gt_idx)
            print(f"  [{model_idx:2d}] {model_name:20s} -> GT[{gt_idx:2d}]")
        else:
            missing_names.append(model_name)
            mapping_indices.append(-1)  # 使用 -1 标记缺失
            print(f"  [{model_idx:2d}] {model_name:20s} -> MISSING")
    
    if missing_names:
        print(f"\n⚠️  Warning: {len(missing_names)} blendshapes not found in mesh:")
        for name in missing_names:
            print(f"    - {name}")
    
    # 检查是否有未使用的 GT blendshapes
    used_gt_indices = set(idx for idx in mapping_indices if idx != -1)
    unused_gt = [name for idx, name in enumerate(TEMPLATE_BS_LIST) if idx not in used_gt_indices]
    if unused_gt:
        print(f"\n📝 Note: {len(unused_gt)} GT blendshapes not used:")
        for name in unused_gt:
            print(f"    - {name}")
    
    return np.array(mapping_indices)

def remap_weights(weights, mapping_indices):
    T, num_bs = weights.shape
    remapped = np.zeros((T, num_bs), dtype=weights.dtype)
    
    for model_idx, gt_idx in enumerate(mapping_indices):
        if gt_idx != -1:  # 跳过缺失的 blendshape
            remapped[:, gt_idx] = weights[:, model_idx]
    
    return remapped

def load_mesh_data(template_path):
    data = np.load(template_path)
    
    print("NPZ file keys:", list(data.keys()))
    print(f"meanshape shape: {data['meanshape'].shape}")
    print(f"faces shape: {data['faces'].shape}")
    print(f"blendshape shape: {data['blendshape'].shape}")
    print(f"bs_names: {data['bs_names'][:5]}...")
    
    meanshape = torch.from_numpy(data['meanshape']).float()
    faces = torch.from_numpy(data['faces']).long()

    # faces可能是1-indexed，需要转换为0-indexed
    if faces.min() == 1:
        faces = faces - 1
    
    # blendshape已经是差值形式
    blendshape = torch.from_numpy(data['blendshape']).float()
    TEMPLATE_BS_LIST = data['bs_names']
    
    print(f"\nMesh info:")
    print(f"  Vertices: {meanshape.shape[0]}")
    print(f"  Faces: {faces.shape[0]}")
    print(f"  Blendshapes: {blendshape.shape[0]}")
    print(f"  Vertex range: X[{meanshape[:, 0].min():.3f}, {meanshape[:, 0].max():.3f}], "
          f"Y[{meanshape[:, 1].min():.3f}, {meanshape[:, 1].max():.3f}], "
          f"Z[{meanshape[:, 2].min():.3f}, {meanshape[:, 2].max():.3f}]")
    
    return meanshape, faces, blendshape, TEMPLATE_BS_LIST

def apply_blendshapes(meanshape, blendshape, weights):
    weights = weights.reshape(52, 1, 1)  # (52, 1, 1)
    weighted_deltas = (blendshape * weights).sum(dim=0)  # (V, 3)
    return weighted_deltas + meanshape

def setup_renderer(meanshape, image_size=(512, 512)):
    # 计算网格边界
    x_max = y_max = meanshape[..., 0:2].abs().max()
    
    # 设置旋转矩阵（镜像变换）
    R = torch.eye(3, device=meanshape.device)
    R[0, 0] = -1  # X轴镜像
    R[2, 2] = -1  # Z轴镜像
    R = R.unsqueeze(0)
    
    # 相机位置
    T = torch.zeros(3, device=meanshape.device)
    T[2] = 10  # 相机后退
    T = T.unsqueeze(0)
    
    # 正交相机
    cameras = FoVOrthographicCameras(
        device=meanshape.device,
        R=R,
        T=T,
        znear=0.01,
        zfar=3,
        max_x=x_max * 1.2,
        min_x=-x_max * 1.2,
        max_y=y_max * 1.2,
        min_y=-y_max * 1.2,
    )
    
    # 光栅化设置
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    # 方向光
    lights = DirectionalLights(
        device=meanshape.device,
        direction=((0, 0, 1),),
        ambient_color=((0.3, 0.3, 0.3),),
        diffuse_color=((0.6, 0.6, 0.6),),
        specular_color=((0.1, 0.1, 0.1),)
    )
    
    # 材质
    materials = Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=15,
        device=meanshape.device
    )
    
    # 混合参数
    blend_params = BlendParams(
        sigma=0.0,
        gamma=0.0,
        background_color=(0.0, 0.0, 0.0)
    )
    
    # 着色器
    shader = HardPhongShader(
        device=meanshape.device,
        cameras=cameras,
        lights=lights,
        materials=materials,
        blend_params=blend_params
    )
    
    # 渲染器
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )
    
    return renderer

def render_frame(renderer, verts, faces):
    # 白色纹理
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb)
    
    # 创建mesh
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )
    
    # 渲染
    with torch.no_grad():
        images = renderer(mesh)
    
    return images[0, ..., :3]  # (H, W, 3)


def images_to_video(image_array, output_path, fps=30, audio_input=None):
    T, H, W, C = image_array.shape
    assert C == 3, "Expected RGB images"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if H % 2 != 0:
        image_array = np.pad(image_array, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant')
        H += 1
    if W % 2 != 0:
        image_array = np.pad(image_array, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant')
        W += 1

    image_array_bgr = image_array[..., ::-1]  # RGB -> BGR

    command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',  # 从 stdin 输入图像
    ]

    if audio_input is not None:
        command += ['-i', audio_input, '-map', '0:v', '-map', '1:a', '-c:a', 'aac']  # 自动截断到较短流
    else:
        command += ['-an']  # 无音频

    command += [
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ]

    print(f"Resolution: {W}x{H}, FPS: {fps}, Frames: {T}")
    if audio_input is not None:
        print(f"Audio input provided (will be synchronized)")
    print(f"Command: {' '.join(command)}")

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        all_frames = image_array_bgr.tobytes()
        stdout, stderr = process.communicate(input=all_frames, timeout=600)
    except subprocess.TimeoutExpired:
        process.kill()
        stderr = process.stderr.read()
        raise RuntimeError(f"FFmpeg 超时\n{stderr.decode()}") from None
    except Exception as e:
        process.kill()
        raise RuntimeError(f"写入视频时发生错误: {e}") from None

    if process.returncode != 0:
        error_msg = stderr.decode()
        print(f"FFmpeg 错误输出:\n{error_msg}")
        raise RuntimeError(f"FFmpeg 编码失败 (返回码 {process.returncode})\n命令: {' '.join(command)}")