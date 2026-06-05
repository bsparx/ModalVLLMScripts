import socket
import subprocess
import time
import modal

MODEL_NAME = "k2-fsa/OmniVoice"
MODEL_PATH = "/model"

def download_model():
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_PATH,
        ignore_patterns=["*.pt", "*.bin"],
    )

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.19.0",
        "transformers>=4.57.0,<5",  # Use >=5.3.0 if you need voice cloning
        "requests",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "TRITON_CACHE_DIR": "/tmp/triton",
            "TORCH_NCCL_ENABLE_MONITORING": "0",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_DEBUG": "OFF",
            "VLLM_HOST_IP": "127.0.0.1",
        }
    )
    .run_function(download_model, secrets=[modal.Secret.from_name("hf-secret")])
)

app = modal.App("omnivoice-tts-inference")

cache_vol = modal.Volume.from_name("vllm-omnivoice-cache", create_if_missing=True)

VLLM_PORT = 8091
MINUTES = 60


@app.cls(
    image=vllm_image,
    gpu="T4",
    scaledown_window=300,
    timeout=40 * MINUTES,
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/cache": cache_vol},
)
@modal.concurrent(max_inputs=20)
class OmniVoiceServer:
    def wait_ready(self):
        """Wait until vLLM server is responsive."""
        while True:
            try:
                socket.create_connection(("127.0.0.1", VLLM_PORT), timeout=1).close()
                return
            except OSError:
                if self.process.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {self.process.returncode}")
                time.sleep(1)

    @modal.enter(snap=True)
    def start(self):
        import requests

        cmd = [
            "vllm",
            "serve",
            MODEL_PATH,
            "--served-model-name",
            MODEL_NAME,
            "--omni",                          # Required for OmniVoice
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--gpu-memory-utilization",
            "0.9",                             # Recommended default for OmniVoice
            "--trust-remote-code",
            "--disable-log-stats",
            "--enable-sleep-mode",
        ]

        print("Starting vLLM OmniVoice server...")
        self.process = subprocess.Popen(cmd)

        self.wait_ready()
        print("vLLM is up! Warming up...")

        # Warmup with a basic TTS request
        warmup_payload = {
            "model": MODEL_NAME,
            "input": "Hello, how are you?",
            "voice": "default",
            "response_format": "wav",
        }
        try:
            requests.post(
                f"http://127.0.0.1:{VLLM_PORT}/v1/audio/speech",
                json=warmup_payload,
                timeout=300,
            ).raise_for_status()
            print("Warmup complete!")
        except Exception as e:
            print(f"Warmup failed: {e}")

        print("Putting vLLM to sleep for snapshotting...")
        requests.post(f"http://127.0.0.1:{VLLM_PORT}/sleep?level=1").raise_for_status()

    @modal.enter(snap=False)
    def wake_up(self):
        import requests

        print("Waking up vLLM from snapshot...")
        requests.post(f"http://127.0.0.1:{VLLM_PORT}/wake_up").raise_for_status()
        self.wait_ready()
        print("vLLM is awake and ready!")

    @modal.exit()
    def stop(self):
        if hasattr(self, "process"):
            self.process.terminate()

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass