import argparse
import html
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torchaudio

from ex_omni.constants import SPEECH_TOKEN_INDEX
from ex_omni.flow_inference import AudioDecoder
from ex_omni.model.builder import load_model_for_inference
from ex_omni.render_utils import (
    MODEL_BS_LIST,
    apply_blendshapes,
    create_blendshape_mapping,
    images_to_video,
    load_mesh_data,
    remap_weights,
    render_frame,
    setup_renderer,
)


temp_dir = Path(os.environ.get("EX_OMNI_TEMP_DIR", "./outputs"))
temp_dir.mkdir(parents=True, exist_ok=True)


def setup_logging():
    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(f"logs/gradio_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("ExOmni")
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logging()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.lock = threading.RLock()

    def create_session(self, session_id: str):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "history": [],
                    "last_active": time.time(),
                    "request_count": 0,
                }

    def get_history(self, session_id: str):
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["last_active"] = time.time()
                return self.sessions[session_id]["history"]
            return []

    def update_history(self, session_id: str, history: list):
        with self.lock:
            if session_id not in self.sessions:
                self.create_session(session_id)
            self.sessions[session_id]["history"] = history
            self.sessions[session_id]["last_active"] = time.time()
            self.sessions[session_id]["request_count"] += 1

    def clear_session(self, session_id: str):
        with self.lock:
            self.sessions.pop(session_id, None)

    def cleanup_old_sessions(self, timeout: int = 3600):
        with self.lock:
            current_time = time.time()
            expired = [
                sid
                for sid, data in self.sessions.items()
                if current_time - data["last_active"] > timeout
            ]
            for sid in expired:
                del self.sessions[sid]
            return len(expired)

    def get_session_count(self):
        with self.lock:
            return len(self.sessions)


class MultiUserChatbot:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, args):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, args):
        if MultiUserChatbot._initialized:
            logger.warning("Model is already initialized, skip reloading.")
            return

        self.args = args
        self.session_manager = SessionManager()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ExOmniWorker")
        self.generation_lock = threading.Lock()
        self.processing_requests = set()
        self.request_lock = threading.Lock()
        self.renderer = None
        self.meanshape = None
        self.faces = None
        self.blendshape = None
        self.TEMPLATE_BS_LIST = None
        self.mapping_indices = None
        self.render_batch_size = 10

        logger.info("=" * 60)
        logger.info("Initializing Ex-Omni inference system...")
        logger.info("=" * 60)

        logger.info("Loading main model...")
        self.tokenizer, self.model = load_model_for_inference(
            os.path.expanduser(args.model_path),
            load_bf16=args.load_bf16,
            device_map="auto",
            attn_implementation=args.attn_implementation,
        )
        self.model.eval()
        logger.info("Main model loaded.")

        logger.info("Initializing audio decoder...")
        try:
            self.audio_decoder = AudioDecoder(
                config_path="./cosyvoice/vocab_16K.yaml",
                flow_ckpt_path=args.flow_ckpt_path,
                hift_ckpt_path=args.hift_ckpt_path,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )
            logger.info("Audio decoder ready.")
        except Exception as exc:
            logger.warning(f"Audio decoder failed to load: {exc}")
            self.audio_decoder = None

        logger.info("Loading render assets...")
        self.meanshape, self.faces, self.blendshape, self.TEMPLATE_BS_LIST = load_mesh_data(args.template_path)
        render_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.meanshape = self.meanshape.to(render_device)
        self.faces = self.faces.to(render_device)
        self.blendshape = self.blendshape.to(render_device)
        self.mapping_indices = create_blendshape_mapping(self.TEMPLATE_BS_LIST, self.TEMPLATE_BS_LIST)
        self.renderer = setup_renderer(self.meanshape, image_size=(args.render_height, args.render_width))
        logger.info("Render assets ready.")

        self.tokenizer.add_tokens(["<speech>"], special_tokens=True)
        self.tokenizer.chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
            "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        self.speech_token_index = self.tokenizer.convert_tokens_to_ids("<speech>")
        self.system_message = (
            "You are a multimodal assistant that understands both speech and text, "
            "and can respond using natural language or synthesized speech."
        )
        self._start_cleanup_task()
        MultiUserChatbot._initialized = True
        logger.info("Ex-Omni inference system initialized.")

    def _start_cleanup_task(self):
        def cleanup_loop():
            while True:
                time.sleep(600)
                try:
                    self.session_manager.cleanup_old_sessions(timeout=3600)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    logger.error(f"Cleanup failed: {exc}")

        threading.Thread(target=cleanup_loop, daemon=True, name="CleanupThread").start()

    def get_or_create_session_history(self, session_id: str):
        history = self.session_manager.get_history(session_id)
        if not history:
            history = [{"role": "system", "content": self.system_message}]
            self.session_manager.update_history(session_id, history)
        return history

    def process_speech(self, speech_file: str) -> Tuple[torch.Tensor, int, str]:
        waveform, sr = torchaudio.load(speech_file)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        audio_array = waveform.squeeze().numpy()
        if audio_array is None or np.isnan(audio_array).any():
            raise ValueError(f"Error loading speech: {speech_file}")

        features = self.model.get_model().speech_encoder.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )
        speech_encoder = self.model.get_model().speech_encoder
        encoder_device = next(speech_encoder.parameters()).device
        encoder_dtype = next(speech_encoder.parameters()).dtype
        input_features = features["input_features"].to(
            dtype=encoder_dtype,
            device=encoder_device,
            non_blocking=True,
        )
        return input_features, input_features.shape[-1], speech_file

    def save_blendshape_data(self, bs_pred: np.ndarray, session_id: str, timestamp: int) -> Tuple[str, str]:
        facs_names = [
            "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
            "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
            "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
            "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
            "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
            "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
            "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
            "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
            "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
            "tongueOut",
        ]
        data = {
            "exportFps": 30,
            "trackPath": f"animation_track_{session_id[:8]}_{timestamp}",
            "numPoses": len(facs_names),
            "numFrames": bs_pred.shape[0],
            "facsNames": facs_names,
            "weightMat": bs_pred.tolist(),
        }
        base_path = temp_dir / f"blendshape_{session_id}_{timestamp}"
        json_path = str(base_path.with_suffix(".json"))
        npy_path = str(base_path.with_suffix(".npy"))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        np.save(npy_path, bs_pred)
        return json_path, npy_path

    def generate_response(
        self,
        session_id: str,
        user_input: str,
        speech_file: Optional[str] = None,
        render_video_output: bool = True,
    ) -> Dict[str, Any]:
        timestamp = int(time.time())
        conversation_history = self.get_or_create_session_history(session_id)

        if speech_file:
            generation_message = f"<speech>\n{user_input}" if user_input else "<speech>"
            display_message = f"[Speech] {user_input}" if user_input else "[Speech input]"
        else:
            generation_message = user_input
            display_message = user_input

        temp_history = conversation_history + [{"role": "user", "content": generation_message}]
        input_id = self.tokenizer.apply_chat_template(temp_history, add_generation_prompt=True)
        if not input_id:
            raise ValueError("Tokenizer returned empty input ids.")

        for idx, encode_id in enumerate(input_id):
            if encode_id == self.speech_token_index:
                input_id[idx] = SPEECH_TOKEN_INDEX

        model_device = next(self.model.parameters()).device
        input_ids = torch.tensor([input_id], dtype=torch.long, device=model_device)

        if speech_file:
            speech_tensor, speech_length, _ = self.process_speech(speech_file)
            speech_length = torch.LongTensor([speech_length])
        else:
            speech_encoder = self.model.get_model().speech_encoder
            encoder_device = next(speech_encoder.parameters()).device
            encoder_dtype = next(speech_encoder.parameters()).dtype
            mel_size = getattr(self.args, "mel_size", 128)
            speech_tensor = torch.zeros(1, mel_size, 3000, dtype=encoder_dtype, device=encoder_device)
            speech_length = torch.LongTensor([3000])

        with self.generation_lock:
            self.model.eval()
            with torch.inference_mode():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                time1 = time.time()
                do_sample = self.args.temperature > 0
                generation_kwargs = {
                    "speech": speech_tensor,
                    "speech_lengths": speech_length,
                    "do_sample": do_sample,
                    "num_beams": self.args.num_beams,
                    "max_new_tokens": self.args.max_new_tokens,
                    "use_cache": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "streaming_unit_gen": False,
                    "faster_infer": False if self.args.s2s else True,
                }
                if do_sample:
                    generation_kwargs["temperature"] = self.args.temperature
                    if self.args.top_p is not None:
                        generation_kwargs["top_p"] = self.args.top_p
                output_ids, output_units, bs_pred = self.model.generate(
                    input_ids,
                    **generation_kwargs,
                )
                time2 = time.time()

        text_response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        conversation_history.append({"role": "user", "content": display_message})
        conversation_history.append({"role": "assistant", "content": text_response})
        if self.args.max_history_length > 0 and len(conversation_history) > self.args.max_history_length * 2 + 1:
            conversation_history = [conversation_history[0]] + conversation_history[-(self.args.max_history_length * 2):]
        self.session_manager.update_history(session_id, conversation_history)

        audio_path = None
        if self.args.s2s and output_units is not None and self.audio_decoder:
            try:
                output_units_list = [int(x) for x in str(output_units).split()]
                tts_token = torch.tensor(output_units_list, device="cuda:0" if torch.cuda.is_available() else "cpu").unsqueeze(0)
                tts_speech = self.audio_decoder.offline_inference(tts_token)
                audio_path = str(temp_dir / f"response_{session_id}_{timestamp}.wav")
                torchaudio.save(audio_path, tts_speech.cpu(), sample_rate=22050, format="wav")
            except Exception as exc:
                logger.warning(f"Speech synthesis failed: {exc}")

        video_path = None
        blendshape_json_path = None
        blendshape_npy_path = None
        if bs_pred is not None and self.args.save_blendshape and self.renderer:
            try:
                bs_pred = bs_pred.squeeze(0).detach().to(torch.float32).cpu().numpy()
                if bs_pred.shape[-1] == 51:
                    bs_pred = np.pad(bs_pred, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
                bs_pred = np.clip(bs_pred, 0, 1)
                blendshape_json_path, blendshape_npy_path = self.save_blendshape_data(bs_pred, session_id, timestamp)
                if render_video_output:
                    video_path = self.render_video(bs_pred, audio_path, timestamp)
            except Exception:
                logger.warning("Video generation failed:\n" + traceback.format_exc())

        return {
            "text": text_response,
            "audio_path": audio_path,
            "video_path": video_path,
            "blendshape_json": blendshape_json_path,
            "blendshape_npy": blendshape_npy_path,
            "time": time2 - time1,
            "timestamp": timestamp,
        }

    def generate_response_with_timeout(
        self,
        session_id: str,
        user_input: str,
        speech_file: Optional[str] = None,
        timeout: int = 120,
        render_video_output: bool = True,
    ) -> Dict[str, Any]:
        request_id = f"{session_id}_{hash(user_input)}_{hash(speech_file) if speech_file else 0}_{int(render_video_output)}"
        with self.request_lock:
            if request_id in self.processing_requests:
                raise ValueError("Request is already being processed.")
            self.processing_requests.add(request_id)
        try:
            future = self.executor.submit(
                self.generate_response,
                session_id,
                user_input,
                speech_file,
                render_video_output,
            )
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"Generation timed out after {timeout}s")
        finally:
            with self.request_lock:
                self.processing_requests.discard(request_id)

    def render_video(self, bs_pred, audio_path=None, timestamp=None):
        if timestamp is None:
            timestamp = int(time.time())
        bs_weights = remap_weights(bs_pred, self.mapping_indices)
        bs_weights = torch.from_numpy(bs_weights).float().to(self.meanshape.device)
        rendered_images = []
        for i in range(bs_weights.shape[0]):
            deformed_verts = apply_blendshapes(self.meanshape, self.blendshape, bs_weights[i])
            image = render_frame(self.renderer, deformed_verts, self.faces)
            rendered_images.append((image.detach().cpu().numpy() * 255).astype(np.uint8))
        video_path = str(temp_dir / f"video_{timestamp}.mp4")
        images_to_video(np.stack(rendered_images, axis=0), video_path, fps=30, audio_input=audio_path)
        return video_path

    def clear_session(self, session_id: str):
        self.session_manager.clear_session(session_id)

    def get_stats(self) -> Dict[str, Any]:
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        return {
            "active_sessions": self.session_manager.get_session_count(),
            "gpu_count": torch.cuda.device_count(),
            "initialized": MultiUserChatbot._initialized,
            "processing_requests": len(self.processing_requests),
            "renderer_type": "base",
            "gpu_memory_gb": f"{gpu_mem:.2f}",
        }


chatbot_instance = None


def normalize_chat_history(history: Optional[list]) -> list[dict[str, Any]]:
    normalized_history = []
    for message in history or []:
        if isinstance(message, dict):
            normalized_history.append(message)
        elif isinstance(message, (list, tuple)) and len(message) == 2:
            user_message, assistant_message = message
            if user_message is not None:
                normalized_history.append({"role": "user", "content": user_message})
            if assistant_message is not None:
                normalized_history.append({"role": "assistant", "content": assistant_message})
    return normalized_history


def _copy_user_audio_to_temp(audio_path: Optional[str], session_id: str) -> Optional[str]:
    if not audio_path:
        return audio_path
    ext = os.path.splitext(audio_path)[1] or ".wav"
    copied_path = temp_dir / f"user_{session_id}_{int(time.time() * 1000)}{ext}"
    try:
        shutil.copy(audio_path, copied_path)
        return str(copied_path)
    except Exception as exc:
        logger.warning(f"Failed to copy user audio into temp_dir, fallback to original path: {exc}")
        return audio_path


def build_avatar_stage_html(stage: str, detail: Optional[str] = None) -> str:
    stage_meta = {
        "idle": ("Avatar standby", "Facial animation engine is ready for the next turn.", "#7c9cff"),
        "thinking": ("Multimodal reasoning", "Aligning text, speech intent, and facial motion priors.", "#8b5cf6"),
        "rendering": ("Avatar synthesis", "Voice is ready. Composing facial dynamics and final frames.", "#22c55e"),
        "error": ("Avatar unavailable", "This turn could not produce an avatar preview.", "#ef4444"),
    }
    title, subtitle, accent = stage_meta.get(stage, stage_meta["thinking"])
    if detail:
        subtitle = detail
    title = html.escape(title)
    subtitle = html.escape(subtitle)
    return f"""
    <div style=\"min-height: 320px; border-radius: 18px; overflow: hidden; position: relative; border: 1px solid rgba(255,255,255,0.10); background: radial-gradient(circle at top, rgba(255,255,255,0.08), rgba(15,23,42,0.96) 58%);\">
      <style>
        @keyframes exomni_avatar_spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        @keyframes exomni_avatar_pulse {{ 0%, 100% {{ transform: scale(0.92); opacity: 0.55; }} 50% {{ transform: scale(1.08); opacity: 1; }} }}
        @keyframes exomni_avatar_wave {{ 0%, 100% {{ transform: scaleY(0.35); opacity: 0.35; }} 50% {{ transform: scaleY(1); opacity: 1; }} }}
      </style>
      <div style=\"position:absolute; inset:0; background: radial-gradient(circle at 50% 30%, {accent}33 0%, transparent 48%);\"></div>
      <div style=\"position:absolute; inset:0; backdrop-filter: blur(4px);\"></div>
      <div style=\"position:relative; z-index:1; min-height:320px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; padding:26px; text-align:center; color:white;\">
        <div style=\"position:relative; width:88px; height:88px;\">
          <div style=\"position:absolute; inset:0; border-radius:9999px; border:2px solid {accent}; border-top-color:transparent; animation: exomni_avatar_spin 2.8s linear infinite;\"></div>
          <div style=\"position:absolute; inset:14px; border-radius:9999px; background:{accent}; filter: blur(16px); animation: exomni_avatar_pulse 1.8s ease-in-out infinite;\"></div>
          <div style=\"position:absolute; inset:24px; border-radius:9999px; background:linear-gradient(135deg, {accent}, #ffffff); box-shadow: 0 0 28px {accent}88;\"></div>
        </div>
        <div style=\"font-size:20px; font-weight:600; letter-spacing:0.01em;\">{title}</div>
        <div style=\"max-width:280px; font-size:13px; line-height:1.6; color:rgba(255,255,255,0.72);\">{subtitle}</div>
        <div style=\"display:flex; align-items:flex-end; gap:8px; height:30px;\">
          <span style=\"width:7px; height:100%; border-radius:9999px; background:{accent}; animation: exomni_avatar_wave 1s ease-in-out infinite; animation-delay:0s;\"></span>
          <span style=\"width:7px; height:100%; border-radius:9999px; background:{accent}; animation: exomni_avatar_wave 1s ease-in-out infinite; animation-delay:0.12s;\"></span>
          <span style=\"width:7px; height:100%; border-radius:9999px; background:{accent}; animation: exomni_avatar_wave 1s ease-in-out infinite; animation-delay:0.24s;\"></span>
          <span style=\"width:7px; height:100%; border-radius:9999px; background:{accent}; animation: exomni_avatar_wave 1s ease-in-out infinite; animation-delay:0.36s;\"></span>
        </div>
      </div>
    </div>
    """


def prepare_user_message(text_input: str, audio_input: Optional[str], history: list, session_state: dict) -> Tuple:
    history = normalize_chat_history(history)

    if session_state is None or "session_id" not in session_state:
        session_id = str(uuid.uuid4())
        session_state = {"session_id": session_id}
    else:
        session_id = session_state["session_id"]

    if not text_input and not audio_input:
        error_msg = "Please input text or upload audio."
        return (
            history,
            None,
            gr.update(visible=False, value=None),
            gr.update(value=build_avatar_stage_html("idle"), visible=True),
            None,
            None,
            error_msg,
            session_state,
            None,
            None,
        )

    audio_input_local = _copy_user_audio_to_temp(audio_input, session_id) if audio_input else None
    if text_input and audio_input_local:
        input_type = "text+audio"
        user_content: Any = [text_input, gr.Audio(value=audio_input_local, autoplay=False)]
    elif audio_input_local:
        input_type = "audio"
        user_content = gr.Audio(value=audio_input_local, autoplay=False)
    else:
        input_type = "text"
        user_content = text_input

    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": "Generating..."})

    pending_request = {
        "session_id": session_id,
        "text_input": text_input or "",
        "audio_input": audio_input_local,
        "input_type": input_type,
    }
    status = f"Queued | Input: {input_type} | Session: {session_id[:8]}"
    return (
        history,
        None,
        gr.update(visible=False, value=None),
        gr.update(value=build_avatar_stage_html("thinking"), visible=True),
        None,
        None,
        status,
        session_state,
        pending_request,
        None,
    )


def generate_assistant_response(pending_request: Optional[dict], history: list, session_state: dict) -> Tuple:
    global chatbot_instance
    history = normalize_chat_history(history)

    if not pending_request:
        return (
            history,
            None,
            gr.update(value=build_avatar_stage_html("idle"), visible=True),
            None,
            None,
            "No pending request.",
            session_state,
            None,
        )

    if chatbot_instance is None or not MultiUserChatbot._initialized:
        error_msg = "Model is not initialized. Please check server logs."
        if history and history[-1].get("role") == "assistant":
            history[-1] = {"role": "assistant", "content": error_msg}
        else:
            history.append({"role": "assistant", "content": error_msg})
        return (
            history,
            None,
            gr.update(value=build_avatar_stage_html("error", error_msg), visible=True),
            None,
            None,
            error_msg,
            session_state,
            None,
        )

    session_id = pending_request["session_id"]
    text_input = pending_request.get("text_input", "")
    audio_input = pending_request.get("audio_input")
    input_type = pending_request.get("input_type", "text")

    try:
        response = chatbot_instance.generate_response_with_timeout(
            session_id=session_id,
            user_input=text_input,
            speech_file=audio_input,
            timeout=360,
            render_video_output=False,
        )
        audio_path = response.get("audio_path")
        assistant_content: Any = response["text"]
        if audio_path:
            assistant_content = [response["text"], gr.Audio(value=audio_path, autoplay=False)]

        if history and history[-1].get("role") == "assistant":
            history[-1] = {"role": "assistant", "content": assistant_content}
        else:
            history.append({"role": "assistant", "content": assistant_content})

        stats = chatbot_instance.get_stats()
        status = (
            f"Response ready in {response['time']:.2f}s | Input: {input_type} | "
            f"Session: {session_id[:8]} | Active: {stats['active_sessions']} | GPU: {stats['gpu_memory_gb']} GB"
        )
        pending_response = {
            **response,
            "session_id": session_id,
            "input_type": input_type,
        }
        return (
            history,
            audio_path,
            gr.update(value=build_avatar_stage_html("rendering"), visible=True),
            response.get("blendshape_json"),
            response.get("blendshape_npy"),
            status,
            session_state,
            pending_response,
        )
    except Exception as exc:
        logger.error(traceback.format_exc())
        status = f"Generation failed: {exc}"
        if history and history[-1].get("role") == "assistant":
            history[-1] = {"role": "assistant", "content": status}
        else:
            history.append({"role": "assistant", "content": status})
        return (
            history,
            None,
            gr.update(value=build_avatar_stage_html("error", status), visible=True),
            None,
            None,
            status,
            session_state,
            None,
        )


def render_pending_video(pending_response: Optional[dict], current_status: str) -> Tuple:
    global chatbot_instance

    if not pending_response:
        return gr.update(visible=False, value=None), gr.update(value=build_avatar_stage_html("idle"), visible=True), current_status, None

    if chatbot_instance is None or not MultiUserChatbot._initialized or not chatbot_instance.renderer:
        status = f"{current_status} | Video renderer unavailable."
        return (
            gr.update(visible=False, value=None),
            gr.update(value=build_avatar_stage_html("error", status), visible=True),
            status,
            None,
        )

    blendshape_npy = pending_response.get("blendshape_npy")
    if not blendshape_npy or not os.path.exists(blendshape_npy):
        status = f"{current_status} | Avatar preview unavailable for this turn."
        return (
            gr.update(visible=False, value=None),
            gr.update(value=build_avatar_stage_html("error", status), visible=True),
            status,
            None,
        )

    session_id = pending_response.get("session_id", "unknown")
    input_type = pending_response.get("input_type", "text")

    try:
        bs_pred = np.load(blendshape_npy)
        video_path = chatbot_instance.render_video(
            bs_pred,
            pending_response.get("audio_path"),
            pending_response.get("timestamp"),
        )
        stats = chatbot_instance.get_stats()
        status = (
            f"Video ready | Input: {input_type} | Session: {session_id[:8]} | "
            f"Active: {stats['active_sessions']} | GPU: {stats['gpu_memory_gb']} GB"
        )
        return (
            gr.update(value=video_path, visible=True),
            gr.update(value=build_avatar_stage_html("idle"), visible=False),
            status,
            None,
        )
    except Exception:
        logger.warning("Video generation failed:\n" + traceback.format_exc())
        status = f"{current_status} | Video generation failed."
        return (
            gr.update(visible=False, value=None),
            gr.update(value=build_avatar_stage_html("error", status), visible=True),
            status,
            None,
        )


def clear_conversation(session_state: dict) -> Tuple:
    global chatbot_instance
    if chatbot_instance and session_state and "session_id" in session_state:
        chatbot_instance.clear_session(session_state["session_id"])
    return (
        [],
        None,
        gr.update(visible=False, value=None),
        gr.update(value=build_avatar_stage_html("idle"), visible=True),
        None,
        None,
        "Conversation cleared.",
        session_state,
        None,
        None,
    )


def create_gradio_interface(args):
    with gr.Blocks(title="Ex-Omni Demo") as demo:
        gr.Markdown("# Ex-Omni Demo\n\nText/speech interaction with speech and 3D facial animation generation.")
        session_state = gr.State()
        pending_request_state = gr.State()
        pending_response_state = gr.State()
        status_timer = gr.Timer(value=5, active=True)
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=500, group_consecutive_messages=True)
                with gr.Row():
                    text_input = gr.Textbox(label="Text Input", placeholder="Input your message...", scale=4, lines=11)
                    audio_input = gr.Audio(label="Speech Input", type="filepath", scale=1)
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", size="lg")
            with gr.Column(scale=1):
                status_text = gr.Textbox(label="Status", value="Waiting for model initialization...", interactive=False, lines=3)
                audio_output = gr.Audio(label="Speech Response", type="filepath", interactive=False)
                avatar_stage = gr.HTML(value=build_avatar_stage_html("idle"))
                video_output = gr.Video(label="Avatar Video", interactive=False, visible=False)
                with gr.Accordion("Blendshape Data", open=False):
                    blendshape_json_file = gr.File(label="JSON", interactive=False)
                    blendshape_npy_file = gr.File(label="NPY", interactive=False)

        def update_status():
            if chatbot_instance and MultiUserChatbot._initialized:
                stats = chatbot_instance.get_stats()
                return f"System ready | GPU: {stats['gpu_count']} | Memory: {stats['gpu_memory_gb']} GB | Active: {stats['active_sessions']}"
            return "Model loading..."

        def bind_submit_event(trigger):
            return (
                trigger(
                    fn=prepare_user_message,
                    inputs=[text_input, audio_input, chatbot, session_state],
                    outputs=[
                        chatbot,
                        audio_output,
                        video_output,
                        avatar_stage,
                        blendshape_json_file,
                        blendshape_npy_file,
                        status_text,
                        session_state,
                        pending_request_state,
                        pending_response_state,
                    ],
                    show_progress="hidden",
                    queue=False,
                )
                .then(fn=lambda: (None, None), inputs=[], outputs=[text_input, audio_input])
                .then(
                    fn=generate_assistant_response,
                    inputs=[pending_request_state, chatbot, session_state],
                    outputs=[
                        chatbot,
                        audio_output,
                        avatar_stage,
                        blendshape_json_file,
                        blendshape_npy_file,
                        status_text,
                        session_state,
                        pending_response_state,
                    ],
                    show_progress="hidden",
                )
                .then(
                    fn=render_pending_video,
                    inputs=[pending_response_state, status_text],
                    outputs=[video_output, avatar_stage, status_text, pending_response_state],
                    show_progress="hidden",
                )
            )

        demo.load(fn=update_status, inputs=[], outputs=[status_text], queue=False)
        status_timer.tick(
            fn=update_status,
            inputs=[],
            outputs=[status_text],
            show_progress="hidden",
            queue=False,
        )
        bind_submit_event(submit_btn.click)
        bind_submit_event(text_input.submit)
        clear_btn.click(
            fn=clear_conversation,
            inputs=[session_state],
            outputs=[
                chatbot,
                audio_output,
                video_output,
                avatar_stage,
                blendshape_json_file,
                blendshape_npy_file,
                status_text,
                session_state,
                pending_request_state,
                pending_response_state,
            ],
            queue=False,
        )
    return demo
