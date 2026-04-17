import socket
import subprocess
import time

import modal

MODEL_NAME = "zai-org/GLM-OCR"
MODEL_PATH = "/model"


def download_model():
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_PATH,
        ignore_patterns=["*.pt", "*.bin"],  # Ignore PyTorch state dicts to save space
    )


vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")  # ✨ ADDED: Install git so uv can pull from GitHub
    .pip_install("uv")
    .run_commands(
        # Install the nightly build of vLLM for GLM-OCR
        "uv pip install --system -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly",
        # Install transformers from source for >= 5.0.0 compatibility
        "uv pip install --system git+https://github.com/huggingface/transformers.git",
        "uv pip install --system requests huggingface_hub[hf_transfer]"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "TRITON_CACHE_DIR": "/cache/triton",
            "TORCH_NCCL_ENABLE_MONITORING": "0",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "NCCL_P2P_DISABLE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "NCCL_DEBUG": "OFF",
            "VLLM_HOST_IP": "127.0.0.1",
        }
    )
    .run_function(download_model, secrets=[modal.Secret.from_name("hf-secret")])
)

app = modal.App("example-glm-ocr-mtp-inference")

VLLM_PORT = 8000
MINUTES = 60


@app.cls(
    image=vllm_image,
    gpu="T4",  # Downscaled to cost-effective T4
    scaledown_window=180, 
    timeout=40 * MINUTES,
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=100)
class VllmServer:
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

        cmd =[
            "vllm",
            "serve",
            MODEL_PATH,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            "8192",
            
            # --- MEMORY FIXES FOR T4 ---
            "--gpu-memory-utilization", "0.80",  # Lowered from 0.9 to free up VRAM
            "--max-num-seqs", "16",              # Lowered from default 256 to prevent OOM
            
            # --- T4 Compatibility ---
            "--dtype", "half", # Force float16. T4 does not support bfloat16.
            "--trust-remote-code",
            "--disable-log-stats",
            "--enable-sleep-mode",
            
            # --- GLM-OCR MTP Decoding Flags ---
            "--speculative-config.method", "mtp",
            "--speculative-config.num_speculative_tokens", "1",
        ]

        print("Starting vLLM server...")
        self.process = subprocess.Popen(cmd)

        self.wait_ready()
        print("vLLM is up! Warming up multimodal endpoints...")
        
        # ... rest of your code ...
        self.process = subprocess.Popen(cmd)

        self.wait_ready()
        print("vLLM is up! Warming up multimodal endpoints...")

        # Warmup payload with an image to compile the vision components
        warmup_payload = {
            "model": MODEL_NAME,
            "messages":[
                {
                    "role": "user",
                    "content":[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Text Recognition:"
                        }
                    ]
                }
            ],
            "max_tokens": 5,
            "temperature": 0.0
        }
        try:
            requests.post(
                f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
                json=warmup_payload,
                timeout=300,
            ).raise_for_status()
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
        self.process.terminate()

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass