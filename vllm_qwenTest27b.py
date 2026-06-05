import socket
import subprocess
import time
import modal

# 1. SWAP TO THE LORBUS MODEL
MODEL_NAME = "Lorbus/Qwen3.6-27B-int4-AutoRound"
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
    .apt_install("git")  # <--- ADD THIS LINE HERE
    # Add xxhash which is required by the Genesis patches
    .uv_pip_install(
        "vllm>=0.19.0",
        "transformers>=4.56.0,<5",
        "requests",
        "huggingface_hub[hf_transfer]",
        "xxhash", 
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
            "VLLM_SERVER_DEV_MODE": "1",
            "NCCL_DEBUG": "OFF",
            "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
            # Ensure long context check passes
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", 
        }
    )
    # Your existing patch
    .run_commands(
        'python -c \'import sys; f="/usr/local/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"; c=open(f).read(); c=c.replace("cache_tensor.zero_()", "[t.zero_() for t in cache_tensor if t is not None] if isinstance(cache_tensor, (list, tuple)) else cache_tensor.zero_()"); open(f,"w").write(c)\''
    )
    # 2. APPLY THE TURBOQUANT PATCHES
.run_commands(
        "git clone --depth 1 https://github.com/Sandermage/genesis-vllm-patches.git /opt/genesis-vllm-patches",
        # ADD THIS LINE: Replace 'dist-packages' with 'site-packages' in the script
        "sed -i 's/dist-packages/site-packages/g' /opt/genesis-vllm-patches/patch_genesis_unified.py", 
        "python /opt/genesis-vllm-patches/patch_genesis_unified.py"
    )
    # Note: The blog mentions a custom 'patch_tolist_cudagraph.py'. 
    # If the `.tolist()` bug occurs during warmup, you can disable cudagraphs by setting 
    # "--enforce-eager" (sacrificing some TPS) OR apply a similar inline replace patch here 
    # for vllm/v1/attention/backends/turboquant_attn.py replacing `.tolist()` with integer indexing.
    .run_function(download_model, secrets=[modal.Secret.from_name("hf-secret")])
)

app = modal.App("example-qwen3-6-27b-autoround-inference")

cache_vol = modal.Volume.from_name("vllm-compile-cache5", create_if_missing=True)

VLLM_PORT = 8000
MINUTES = 60

@app.cls(
    image=vllm_image,
    gpu="L4",
    scaledown_window=360,
    timeout=40 * MINUTES,
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/cache": cache_vol},
)
@modal.concurrent(max_inputs=100)
class VllmServer:
    def wait_ready(self):
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
            "--quantization",
            "auto_round",          # Required for the Lorbus model
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "half",
            
            # 3. TURBOQUANT SETTINGS
            "--kv-cache-dtype",
            "turboquant_3bit_nc",  # 3-bit K/V with norm correction
            "--max-model-len",
            "125000",              # Massive 125K context!
            "--gpu-memory-utilization",
            "0.97",                # Pushed slightly higher as per the blog
            
            # 4. CHUNKED PREFILL (Recommended for long context)
            "--enable-chunked-prefill",
            "--no-scheduler-reserve-full-isl",
            "--max-num-batched-tokens",
            "4128",                # Updated alignment

            # 5. MTP SPECULATIVE DECODING
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":3}',
            
            "--mamba-cache-mode",
            "align",
            "--mamba-block-size",
            "16",
            "--block-size",
            "32",
            "--max-num-seqs",
            "8",
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--generation-config",
            "vllm",
            "--disable-custom-all-reduce",
            "--language-model-only", # Remove this if you want Vision enabled!
            "--trust-remote-code",
            "--disable-log-stats",
            "--enable-sleep-mode",
            "--reasoning-parser",
            "qwen3",
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
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