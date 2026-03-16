from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = "VisionAI"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/vision_ai.db"

    # Video ingestion
    frame_sample_interval: int = Field(2, description="Seconds between sampled frames")
    max_concurrent_streams: int = 10
    video_storage_path: str = "./data/videos"
    frame_storage_path: str = "./data/frames"
    detection_images_path: str = Field("./data/detections", description="Directory for authorised/unauthorised person crop images")
    rtsp_open_retries: int = Field(3, description="Retries when opening RTSP stream")
    rtsp_retry_delay_seconds: float = Field(2.0, description="Delay between RTSP open retries")
    rtsp_transport: str = Field("auto", description="RTSP transport: tcp, udp, or auto (try tcp then udp)")
    rtsp_timeout_seconds: float = Field(10.0, description="RTSP socket timeout in seconds (0 = default)")

    # YOLO
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence_threshold: float = 0.5
    yolo_device: str = "cpu"

    # Anomaly detection
    idle_threshold_seconds: int = Field(30, description="Seconds stationary before idle_time event")
    unauthorized_zone_alert: bool = True
    alert_unauthorized_person: bool = True
    shift_deviation_threshold: float = 0.3

    # Video recording
    record_video: bool = Field(True, description="Record camera feed to disk while pipeline runs")
    recording_path: str = Field("./data/recordings", description="Directory for active recordings (by camera name)")
    recording_archive_path: str = Field("./data/recordings/archive", description="Directory to move completed recordings (same structure: camera_name/rec_*.mp4)")
    recording_fps: int = Field(5, description="FPS for recorded video (lower = smaller files)")

    # Employee registration & recognition (ONNX face embedding)
    employee_photos_path: str = Field("./data/employees", description="Directory for employee photos")
    face_embedding_enabled: bool = Field(True, description="Identify persons via ONNX face embedding")
    face_embedding_model_path: str = Field("./models/arcface.onnx", description="Path to ArcFace/FaceNet ONNX model (112x112 input, 512-d output)")
    face_match_threshold: float = Field(0.6, description="Cosine similarity threshold for match (0-1). Raise to 0.65+ to reduce wrong person matches.")
    match_only_employees_on_shift: bool = Field(False, description="If True, only employees on current shift are recognised; others show Unknown. If False, all employees are matched.")
    log_face_matches: bool = Field(False, description="If True, log top face-match similarities to help debug recognition")
    # Optional: OpenCV DNN face detector (Caffe). If not set, uses head region from person bbox.
    face_detector_prototxt: str = Field("", description="Path to deploy.prototxt for face detection")
    face_detector_caffemodel: str = Field("", description="Path to .caffemodel for face detection")

    # Alerts
    alert_cooldown_seconds: int = 300
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_recipients: str = ""

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # GPU
    use_gpu: bool = False
    gpu_device_id: int = 0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def device(self) -> str:
        if self.use_gpu:
            return f"cuda:{self.gpu_device_id}"
        return self.yolo_device

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent


settings = Settings()
