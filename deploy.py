from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import deploy_base
from pytorch3d.renderer import (
    DirectionalLights,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.structures import Meshes, join_meshes_as_scene


DEFAULT_EMOTALK_TEMPLATE_PATH = "asset/EmoTalk.npz"
DEFAULT_CLAIRE_TEMPLATE_PATH = "asset/claire.npz"
DEFAULT_TEMPLATE_TYPE = "claire"
TEMPLATE_PATH_BY_TYPE = {
    "emotalk": DEFAULT_EMOTALK_TEMPLATE_PATH,
    "claire": DEFAULT_CLAIRE_TEMPLATE_PATH,
}


FACS_NAMES_52 = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut",
]


def normalize_name_list(names) -> List[str]:
    result: List[str] = []
    for name in names:
        if isinstance(name, bytes):
            result.append(name.decode("utf-8"))
        else:
            result.append(str(name))
    return result


def load_mesh_template_from_npz(template_path: str):
    data = np.load(template_path, allow_pickle=True)
    required = {"meanshape", "faces", "blendshape", "bs_names"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{template_path} is missing required keys: {sorted(missing)}")

    meanshape = torch.from_numpy(np.asarray(data["meanshape"], dtype=np.float32)).float()
    faces = torch.from_numpy(np.asarray(data["faces"], dtype=np.int64)).long()
    blendshape = torch.from_numpy(np.asarray(data["blendshape"], dtype=np.float32)).float()
    bs_names = normalize_name_list(data["bs_names"])

    if faces.min() == 1:
        faces = faces - 1

    return meanshape, faces, blendshape, bs_names


def get_model_bs_names(num_weights: int) -> List[str]:
    # The model output is expected to be (T, 52), where index 51 is tongueOut.
    if num_weights == len(FACS_NAMES_52):
        return FACS_NAMES_52
    raise ValueError(
        f"Unsupported blendshape width {num_weights}. "
        f"Expected {len(FACS_NAMES_52)} from model output, with tongueOut at index 51."
    )


def remap_weights_to_template(
    weights: np.ndarray,
    mapping_indices: np.ndarray,
    template_bs_count: int,
) -> np.ndarray:
    remapped = np.zeros((weights.shape[0], template_bs_count), dtype=weights.dtype)
    for model_idx, template_idx in enumerate(mapping_indices):
        if template_idx >= 0:
            remapped[:, template_idx] = weights[:, model_idx]
    return remapped


def apply_blendshapes_batch(
    meanshape: torch.Tensor,
    blendshape: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)
    weighted = (blendshape.unsqueeze(0) * weights[:, :, None, None]).sum(dim=1)
    return weighted + meanshape.unsqueeze(0)


def load_claire_render_assets(mesh_template_npz_path: str):
    meanshape, faces, blendshape, bs_names = load_mesh_template_from_npz(mesh_template_npz_path)
    static_parts: List[Dict] = []
    return meanshape, faces, blendshape, bs_names, static_parts


def center_vertices(verts_t: torch.Tensor, static_parts: List[Dict] | None = None):
    mins = verts_t.amin(dim=(0, 1))
    maxs = verts_t.amax(dim=(0, 1))
    static_stack = None
    if static_parts:
        static_stack = torch.cat(
            [
                torch.as_tensor(np.array(part["verts"], copy=True), dtype=torch.float32, device=verts_t.device)
                for part in static_parts
            ],
            dim=0,
        )
        mins = torch.minimum(mins, static_stack.amin(dim=0))
        maxs = torch.maximum(maxs, static_stack.amax(dim=0))

    center = (mins + maxs) * 0.5
    centered = verts_t - center.view(1, 1, 3)
    xy_half_extent = centered[..., :2].abs().amax()

    if static_stack is not None:
        centered_static = static_stack - center.view(1, 3)
        xy_half_extent = torch.maximum(xy_half_extent, centered_static[:, :2].abs().amax())

    return centered, center, xy_half_extent


def build_view_rotation(yaw_deg: float, pitch_deg: float, device: torch.device) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    rot_y = torch.tensor(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=torch.float32,
        device=device,
    )
    rot_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=torch.float32,
        device=device,
    )
    mirror = torch.eye(3, dtype=torch.float32, device=device)
    mirror[0, 0] = -1.0
    mirror[2, 2] = -1.0
    return (rot_x @ rot_y @ mirror).unsqueeze(0)


def setup_claire_renderer(
    xy_half_extent: torch.Tensor,
    image_size: Tuple[int, int],
    bg_color: Tuple[float, float, float],
    camera_mode: str,
    padding: float,
    light_ambient: float,
    light_diffuse: float,
    light_specular: float,
    yaw: float,
    pitch: float,
    camera_distance: float,
    fov: float,
    device: torch.device,
):
    xy_bound = torch.as_tensor(xy_half_extent, dtype=torch.float32, device=device) * float(padding)
    R = build_view_rotation(yaw, pitch, device)

    T = torch.zeros((1, 3), dtype=torch.float32, device=device)
    if camera_mode == "perspective":
        fov_rad = math.radians(float(fov))
        auto_distance = float(xy_bound.detach().cpu()) / max(math.tan(fov_rad * 0.5), 1e-4)
        distance = float(camera_distance) if float(camera_distance) > 0.0 else auto_distance * 1.35
        T[0, 2] = distance
        cameras = FoVPerspectiveCameras(
            device=device,
            R=R,
            T=T,
            znear=0.1,
            zfar=max(distance * 4.0, 100.0),
            fov=float(fov),
        )
    else:
        T[0, 2] = 10.0
        cameras = FoVOrthographicCameras(
            device=device,
            R=R,
            T=T,
            znear=0.01,
            zfar=30.0,
            max_x=xy_bound,
            min_x=-xy_bound,
            max_y=xy_bound,
            min_y=-xy_bound,
        )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = DirectionalLights(
        device=device,
        direction=((0.0, 0.0, 1.0),),
        ambient_color=((float(light_ambient), float(light_ambient), float(light_ambient)),),
        diffuse_color=((float(light_diffuse), float(light_diffuse), float(light_diffuse)),),
        specular_color=((float(light_specular), float(light_specular), float(light_specular)),),
    )
    materials = Materials(
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=((1.0, 1.0, 1.0),),
        shininess=15.0,
        device=device,
    )
    blend_params = BlendParams(
        sigma=0.0,
        gamma=0.0,
        background_color=tuple(float(x) for x in bg_color),
    )
    shader = HardPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        materials=materials,
        blend_params=blend_params,
    )

    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader,
    )


def setup_adaptive_ortho_renderer(
    meanshape: torch.Tensor,
    image_size: Tuple[int, int] = (512, 512),
    padding: float = 1.2,
):
    # Keep the original EmoTalk path unchanged; this helper is only for larger templates
    # such as centered Claire NPZs, where the fixed camera distance in render_utils.py
    # becomes too close and can expose interior nose geometry.
    xy_bound = meanshape[..., 0:2].abs().max() * float(padding)
    z_min = meanshape[..., 2].min()
    z_max = meanshape[..., 2].max()
    z_span = (z_max - z_min).abs()
    camera_distance = max(float(z_span.detach().cpu()) * 2.5, 10.0)
    zfar = max(camera_distance + float(z_span.detach().cpu()) * 4.0, 30.0)

    R = torch.eye(3, device=meanshape.device)
    R[0, 0] = -1.0
    R[2, 2] = -1.0
    R = R.unsqueeze(0)

    T = torch.zeros((1, 3), dtype=torch.float32, device=meanshape.device)
    T[0, 2] = camera_distance

    cameras = FoVOrthographicCameras(
        device=meanshape.device,
        R=R,
        T=T,
        znear=0.01,
        zfar=zfar,
        max_x=xy_bound,
        min_x=-xy_bound,
        max_y=xy_bound,
        min_y=-xy_bound,
    )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = DirectionalLights(
        device=meanshape.device,
        direction=((0.0, 0.0, 1.0),),
        ambient_color=((0.3, 0.3, 0.3),),
        diffuse_color=((0.6, 0.6, 0.6),),
        specular_color=((0.1, 0.1, 0.1),),
    )
    materials = Materials(
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=((1.0, 1.0, 1.0),),
        shininess=15.0,
        device=meanshape.device,
    )
    blend_params = BlendParams(
        sigma=0.0,
        gamma=0.0,
        background_color=(0.0, 0.0, 0.0),
    )
    shader = HardPhongShader(
        device=meanshape.device,
        cameras=cameras,
        lights=lights,
        materials=materials,
        blend_params=blend_params,
    )

    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader,
    ), camera_distance, zfar


def center_static_parts(static_parts: List[Dict], center: torch.Tensor) -> List[Dict]:
    center_np = center.detach().cpu().numpy()
    return [
        {
            "verts": part["verts"] - center_np,
            "faces": part["faces"],
            "color": part["color"],
        }
        for part in static_parts
    ]


def resolve_template_defaults(args):
    args.template_path = TEMPLATE_PATH_BY_TYPE[args.template_type]
    return args


def render_claire_frame(
    renderer,
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_color: Tuple[float, float, float],
    static_parts: List[Dict],
    device: torch.device,
) -> torch.Tensor:
    verts_rgb = torch.tensor(face_color, dtype=torch.float32, device=device).view(1, 1, 3)
    verts_rgb = verts_rgb.repeat(1, verts.shape[0], 1)
    scene_parts = [
        Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex(verts_features=verts_rgb),
        )
    ]

    for part in static_parts:
        static_verts = torch.as_tensor(np.array(part["verts"], copy=True), dtype=torch.float32, device=device)
        static_faces = torch.as_tensor(np.array(part["faces"], copy=True), dtype=torch.int64, device=device)
        static_color = torch.tensor(part["color"], dtype=torch.float32, device=device).view(1, 1, 3)
        static_color = static_color.repeat(1, static_verts.shape[0], 1)
        scene_parts.append(
            Meshes(
                verts=[static_verts],
                faces=[static_faces],
                textures=TexturesVertex(verts_features=static_color),
            )
        )

    mesh = join_meshes_as_scene(scene_parts)
    with torch.no_grad():
        image = renderer(mesh)[0, ..., :3]
    return image


class MultiUserChatbot(deploy_base.MultiUserChatbot):
    def __init__(self, args):
        super().__init__(args)
        self.mapping_cache: Dict[int, np.ndarray] = {}
        self.template_bs_names = normalize_name_list(self.TEMPLATE_BS_LIST)
        self.static_parts: List[Dict] = []
        self.use_centered_renderer = False
        self._setup_template_renderer(args)

    def _setup_template_renderer(self, args):
        xy_half_extent = float(self.meanshape[..., 0:2].abs().max().detach().cpu())
        neutral_verts = self.meanshape.unsqueeze(0)
        _, neutral_center, neutral_extent = center_vertices(neutral_verts, self.static_parts)

        if args.template_type == "emotalk":
            deploy_base.logger.info(
                "EmoTalk模板渲染模式: "
                f"template_type={args.template_type}, template_path={args.template_path}, "
                f"xy_half_extent={xy_half_extent:.4f}"
            )
            return

        self.use_centered_renderer = True
        self.renderer = setup_claire_renderer(
            xy_half_extent=neutral_extent,
            image_size=(args.render_height, args.render_width),
            bg_color=tuple(args.render_bg_color),
            camera_mode=args.render_camera_mode,
            padding=args.render_frame_padding,
            light_ambient=args.render_light_ambient,
            light_diffuse=args.render_light_diffuse,
            light_specular=args.render_light_specular,
            yaw=args.render_yaw,
            pitch=args.render_pitch,
            camera_distance=args.render_camera_distance,
            fov=args.render_fov,
            device=self.meanshape.device,
        )
        deploy_base.logger.info(
            "Claire模板渲染模式: "
            f"template_type={args.template_type}, template_path={args.template_path}, "
            f"xy_half_extent={xy_half_extent:.4f}, center={neutral_center.detach().cpu().numpy()}"
        )

    def _get_mapping_indices(self, num_weights: int) -> np.ndarray:
        if num_weights not in self.mapping_cache:
            model_bs_names = get_model_bs_names(num_weights)
            self.mapping_cache[num_weights] = deploy_base.create_blendshape_mapping(
                self.template_bs_names,
                model_bs_names,
            )
        return self.mapping_cache[num_weights]

    def _remap_bs_pred(self, bs_pred: np.ndarray) -> np.ndarray:
        if bs_pred.ndim != 2:
            raise ValueError(f"Expected bs_pred shaped (T, D), got {bs_pred.shape}")
        mapping_indices = self._get_mapping_indices(bs_pred.shape[1])
        return remap_weights_to_template(bs_pred, mapping_indices, len(self.template_bs_names))

    def render_video(self, bs_pred, audio_path=None, timestamp=None):
        if timestamp is None:
            timestamp = int(time.time())

        bs_pred = np.asarray(bs_pred, dtype=np.float32)
        bs_weights = self._remap_bs_pred(bs_pred)
        bs_weights_t = torch.from_numpy(bs_weights).float().to(self.meanshape.device)

        if self.use_centered_renderer:
            verts_t = apply_blendshapes_batch(self.meanshape, self.blendshape, bs_weights_t)
            centered_verts, center, xy_half_extent = center_vertices(verts_t, self.static_parts)
            centered_static_parts = center_static_parts(self.static_parts, center)

            renderer = setup_claire_renderer(
                xy_half_extent=xy_half_extent,
                image_size=(self.args.render_height, self.args.render_width),
                bg_color=tuple(self.args.render_bg_color),
                camera_mode=self.args.render_camera_mode,
                padding=self.args.render_frame_padding,
                light_ambient=self.args.render_light_ambient,
                light_diffuse=self.args.render_light_diffuse,
                light_specular=self.args.render_light_specular,
                yaw=self.args.render_yaw,
                pitch=self.args.render_pitch,
                camera_distance=self.args.render_camera_distance,
                fov=self.args.render_fov,
                device=self.meanshape.device,
            )

            rendered_images = []
            for frame_idx in range(centered_verts.shape[0]):
                image = render_claire_frame(
                    renderer=renderer,
                    verts=centered_verts[frame_idx],
                    faces=self.faces,
                    face_color=tuple(self.args.render_face_color),
                    static_parts=centered_static_parts,
                    device=self.meanshape.device,
                )
                image_np = (image.clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_images.append(image_np)
        else:
            rendered_images = []
            for frame_idx in range(bs_weights_t.shape[0]):
                deformed_verts = apply_blendshapes_batch(
                    self.meanshape,
                    self.blendshape,
                    bs_weights_t[frame_idx],
                )[0]
                image = deploy_base.render_frame(self.renderer, deformed_verts, self.faces)
                image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_images.append(image_np)

        rendered_images = np.stack(rendered_images, axis=0)
        video_path = str(Path(deploy_base.temp_dir) / f"video_{timestamp}.mp4")
        deploy_base.images_to_video(
            rendered_images,
            video_path,
            fps=self.args.render_fps,
            audio_input=audio_path,
        )
        return video_path


deploy_base.MultiUserChatbot = MultiUserChatbot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ckpt/Ex-Omni")
    parser.add_argument("--flow_ckpt_path", type=str, default="ckpt/glm-4-voice-decoder/flow.pt")
    parser.add_argument("--hift_ckpt_path", type=str, default="ckpt/glm-4-voice-decoder/hift.pt")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speech_generator_type", type=str, default="ar")
    parser.add_argument("--load_bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--save_blendshape", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_history_length", type=int, default=20)
    parser.add_argument("--auto_clear", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--template_type", choices=sorted(TEMPLATE_PATH_BY_TYPE.keys()), default="emotalk")

    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--render-camera-mode", choices=["ortho", "perspective"], default="perspective")
    parser.add_argument("--render-frame-padding", type=float, default=1.08)
    parser.add_argument("--render-yaw", type=float, default=0.0)
    parser.add_argument("--render-pitch", type=float, default=0.0)
    parser.add_argument("--render-camera-distance", type=float, default=150.0)
    parser.add_argument("--render-fov", type=float, default=14.0)
    parser.add_argument("--render-bg-color", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--render-face-color", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--render-light-ambient", type=float, default=0.3)
    parser.add_argument("--render-light-diffuse", type=float, default=0.6)
    parser.add_argument("--render-light-specular", type=float, default=0.1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = resolve_template_defaults(args)

    num_gpus = torch.cuda.device_count()
    deploy_base.logger.info(f"\n{'='*60}\n🎮 检测到 {num_gpus} 张 GPU\n{'='*60}\n")
    deploy_base.logger.info(
        "🧩 模板配置: "
        f"template_type={args.template_type}, template_path={args.template_path}"
    )

    try:
        chatbot_instance = MultiUserChatbot(args)
        deploy_base.chatbot_instance = chatbot_instance
    except Exception:
        deploy_base.logger.error(f"\n{'='*60}\n❌ 模型初始化失败！\n{'='*60}")
        raise

    demo = deploy_base.create_gradio_interface(args)
    demo.queue(max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        max_threads=10,
        show_error=True,
        theme=deploy_base.gr.themes.Soft(),
        allowed_paths=[deploy_base.temp_dir],
    )


if __name__ == "__main__":
    main()
