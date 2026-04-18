import socket
import subprocess
import time
import modal

MODEL_NAME = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
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
        "transformers>=4.56.0,<5",
        "requests",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "TRITON_CACHE_DIR": "/tmp/triton",      # Fixed: moved off volume to prevent snapshot restore failure
            "TORCH_NCCL_ENABLE_MONITORING": "0",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "NCCL_P2P_DISABLE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "NCCL_DEBUG": "OFF",
            "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        }
    )
    # FIX: Patch vLLM to handle Mamba state (lists) during FP8 KV cache initialization on wake-up
    .run_commands(
        'python -c \'import sys; f="/usr/local/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"; c=open(f).read(); c=c.replace("cache_tensor.zero_()", "[t.zero_() for t in cache_tensor if t is not None] if isinstance(cache_tensor, (list, tuple)) else cache_tensor.zero_()"); open(f,"w").write(c)\''
    )
    .run_function(download_model, secrets=[modal.Secret.from_name("hf-secret")])
)

app = modal.App("example-qwen3-6-35b-a3b-awq-inference")

# Persists torch.compile cache across cold starts — saves ~155s per boot
cache_vol = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)

VLLM_PORT = 8000
MINUTES = 60

@app.cls(
    image=vllm_image,
    gpu="L40S",
    scaledown_window=240,
    timeout=40 * MINUTES,
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/cache": cache_vol},
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

        cmd = [
            "vllm",
            "serve",
            MODEL_PATH,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "half",
            "--kv-cache-dtype",
            "fp8",
            "--max-model-len",
            "65536",
            "--gpu-memory-utilization",
            "0.9156",               # Bumped per vLLM recommendation with CUDAGRAPHS estimator enabled
            "--mamba-cache-mode",
            "align",
            "--mamba-block-size",
            "16",                   # Was 8 — reduces 10% KV cache padding waste
            "--max-num-batched-tokens",
            "4096",                 # Must be >= block_size (2096) in mamba align mode
            "--block-size",
            "32",
            "--max-num-seqs",
            "8",
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--generation-config",
            "vllm",                 # Prevents model's generation_config.json from overriding sampling params
            "--disable-custom-all-reduce",
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
            "--language-model-only",
            "--trust-remote-code",
            "--disable-log-stats",
            "--enable-sleep-mode",
            # Removed --speculative-config: was forcing PIECEWISE cuda graph mode downgrade
            # Removed --reasoning-parser: disabled thinking mode
        ]

        print("Starting vLLM server...")
        self.process = subprocess.Popen(cmd)

        self.wait_ready()
        print("vLLM is up! Warming up...")

        warmup_payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
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
        if hasattr(self, "process"):
            self.process.terminate()

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass