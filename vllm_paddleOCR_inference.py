import socket
import subprocess
import time

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "PaddlePaddle/PaddleOCR-VL-1.6"
MODEL_PATH = "/model"
VLLM_PORT = 8000
MINUTES = 60

# ---------------------------------------------------------------------------
# Image — nightly vLLM + transformers >= 5.0 (required by PaddleOCR-VL-1.6)
# ---------------------------------------------------------------------------

def download_model():
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_PATH,
        ignore_patterns=["*.pt", "*.bin"],  # skip PyTorch state dicts
    )


vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        # vLLM nightly — needed until v0.11.1 ships with PaddleOCR-VL support
        "uv pip install --system -U vllm --pre "
        "--extra-index-url https://wheels.vllm.ai/nightly "
        "--extra-index-url https://download.pytorch.org/whl/cu124 "
        "--index-strategy unsafe-best-match",
        # transformers >= 5.0 required for PaddleOCR-VL-1.6
        "uv pip install --system git+https://github.com/huggingface/transformers.git",
        "uv pip install --system requests huggingface_hub[hf_transfer]",
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

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("paddleocr-vl-1.6-vllm")


@app.cls(
    image=vllm_image,
    gpu="A10G",          # A10G recommended: 24 GB VRAM, supports bfloat16
                         # Swap to "T4" if cost is priority; forces --dtype half below
    scaledown_window=180,
    timeout=40 * MINUTES,
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=100)
class PaddleOCRVLServer:

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _wait_ready(self):
        """Poll until the vLLM HTTP server is accepting connections."""
        while True:
            try:
                socket.create_connection(("127.0.0.1", VLLM_PORT), timeout=1).close()
                return
            except OSError:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM process exited unexpectedly "
                        f"(returncode={self.process.returncode})"
                    )
                time.sleep(1)

    def _build_vllm_cmd(self) -> list[str]:
        """
        Build the vLLM serve command.

        Key flags explained
        -------------------
        --trust-remote-code          : PaddleOCR-VL ships custom modelling code
        --max-num-batched-tokens     : 16 384 matches the upstream recipe
        --no-enable-prefix-caching   : OCR tasks reuse images rarely; skip hashing
        --mm-processor-cache-gb 0   : disable multimodal processor cache
        --dtype half                 : safe for T4 (no bfloat16); fine on A10G/A100
        --gpu-memory-utilization 0.90: leave headroom for CUDA kernels
        --max-num-seqs 32            : balance throughput vs peak VRAM
        --served-model-name          : alias avoids "model does not exist" errors
        """
        return [
            "vllm", "serve", MODEL_PATH,
            "--served-model-name", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--trust-remote-code",
            "--max-num-batched-tokens", "16384",
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb", "0",
            "--dtype", "half",               # float16 — safe on T4 and A10G
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "32",
            "--disable-log-stats",
            "--enable-sleep-mode",
        ]

    def _warmup(self):
        """Send a lightweight request so CUDA kernels are compiled before first user call."""
        import requests

        warmup_payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "https://ofasys-multimodal-wlcb-3-toshanghai"
                                    ".oss-accelerate.aliyuncs.com/wpf272043"
                                    "/keepme/image/receipt.png"
                                )
                            },
                        },
                        {"type": "text", "text": "OCR:"},
                    ],
                }
            ],
            "max_tokens": 10,
            "temperature": 0.0,
        }
        try:
            resp = requests.post(
                f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
                json=warmup_payload,
                timeout=300,
            )
            resp.raise_for_status()
            print("Warmup successful.")
        except Exception as exc:
            print(f"Warmup failed (non-fatal): {exc}")

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    @modal.enter(snap=True)
    def start(self):
        import requests

        cmd = self._build_vllm_cmd()
        print("Starting vLLM server with command:")
        print(" ".join(cmd))

        self.process = subprocess.Popen(cmd)
        self._wait_ready()
        print("vLLM is up. Running warmup...")

        self._warmup()

        print("Putting vLLM to sleep for snapshotting...")
        requests.post(f"http://127.0.0.1:{VLLM_PORT}/sleep?level=1").raise_for_status()
        print("Snapshot ready.")

    @modal.enter(snap=False)
    def wake_up(self):
        import requests

        print("Waking vLLM from snapshot...")
        requests.post(f"http://127.0.0.1:{VLLM_PORT}/wake_up").raise_for_status()
        self._wait_ready()
        print("vLLM is awake and ready!")

    @modal.exit()
    def stop(self):
        self.process.terminate()

    # ------------------------------------------------------------------ #
    # Public web endpoint                                                  #
    # ------------------------------------------------------------------ #

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass


# ---------------------------------------------------------------------------
# Local test helper  (run with: modal run paddleocr_vl_16_modal.py)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """
    Quick smoke-test: queries the running server with an OCR task.

    Usage:
        modal run paddleocr_vl_16_modal.py

    For production use, deploy first:
        modal deploy paddleocr_vl_16_modal.py

    Then query the printed web endpoint URL directly via the OpenAI-compatible
    /v1/chat/completions API (see query_example() below).
    """
    import urllib.request
    import json

    server = PaddleOCRVLServer()

    # Task options: "ocr" | "table" | "formula" | "chart" | "spotting" | "seal"
    TASKS = {
        "ocr":     "OCR:",
        "table":   "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart":   "Chart Recognition:",
        "spotting":"Spotting:",
        "seal":    "Seal Recognition:",
    }

    task = "ocr"
    image_url = (
        "https://paddle-model-ecology.bj.bcebos.com"
        "/paddlex/imgs/demo_image/paddleocr_vl_demo.png"
    )

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": TASKS[task]},
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    print("=== PaddleOCR-VL-1.6 output ===")
    print(result["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
# Standalone query helper (use after `modal deploy`)
# ---------------------------------------------------------------------------

def query_example(base_url: str, image_url: str, task: str = "ocr") -> str:
    """
    Call a deployed PaddleOCR-VL-1.6 server via the OpenAI-compatible API.

    Args:
        base_url:  The Modal web endpoint URL, e.g.
                   "https://your-org--paddleocr-vl-1-6-vllm-paddleocrv-serve.modal.run"
        image_url: Publicly accessible image URL to parse.
        task:      One of ocr | table | formula | chart | spotting | seal

    Returns:
        Parsed text output from the model.

    Example:
        from paddleocr_vl_16_modal import query_example
        print(query_example(
            base_url="https://<your-endpoint>.modal.run",
            image_url="https://...",
            task="table",
        ))
    """
    from openai import OpenAI

    TASKS = {
        "ocr":      "OCR:",
        "table":    "Table Recognition:",
        "formula":  "Formula Recognition:",
        "chart":    "Chart Recognition:",
        "spotting": "Spotting:",
        "seal":     "Seal Recognition:",
    }

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{base_url.rstrip('/')}/v1",
        timeout=3600,
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": TASKS[task]},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return response.choices[0].message.content