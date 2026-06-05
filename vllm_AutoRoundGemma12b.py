import socket
import subprocess
import time
import modal

MODEL_NAME = "Intel/gemma-4-12B-it-int4-AutoRound"
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

def write_patched_entrypoint():
    """
    Write a patched vLLM entrypoint that:
      - Registers Gemma4UnifiedAssistantConfig into AutoModel (runs in every process,
        including EngineCore subprocesses, because it is outside if __name__ == '__main__')
      - Only calls vllm main() when this file is the top-level script, NOT when Python
        re-imports it as __main__ inside a spawned child process.
    """
    script = (
        "import sys\n"
        "\n"
        "# Register the assistant draft-model config so MTP speculative decoding works.\n"
        "# This block intentionally runs in ALL processes (parent + spawned children)\n"
        "# because each child re-imports this file and needs the mapping too.\n"
        "try:\n"
        "    from transformers.models.gemma4_unified_assistant"
        ".configuration_gemma4_unified_assistant import Gemma4UnifiedAssistantConfig\n"
        "    from transformers.models.gemma4_unified_assistant"
        ".modeling_gemma4_unified_assistant import Gemma4UnifiedAssistantForCausalLM\n"
        "    from transformers import AutoModel, AutoModelForCausalLM\n"
        "    AutoModel.register(\n"
        "        Gemma4UnifiedAssistantConfig,\n"
        "        Gemma4UnifiedAssistantForCausalLM,\n"
        "        exist_ok=True,\n"
        "    )\n"
        "    AutoModelForCausalLM.register(\n"
        "        Gemma4UnifiedAssistantConfig,\n"
        "        Gemma4UnifiedAssistantForCausalLM,\n"
        "        exist_ok=True,\n"
        "    )\n"
        "    print('Registered Gemma4UnifiedAssistantConfig')\n"
        "except Exception as exc:\n"
        "    print('Warning: Gemma4UnifiedAssistantConfig registration:', exc)\n"
        "\n"
        "# IMPORTANT: guard main() so it only runs in the parent process.\n"
        "# vLLM uses VLLM_WORKER_MULTIPROC_METHOD=spawn, which re-imports __main__\n"
        "# in every child. Without this guard, each child tries to start vLLM again,\n"
        "# triggering: 'An attempt has been made to start a new process before\n"
        "# the current process has finished its bootstrapping phase.'\n"
        "if __name__ == '__main__':\n"
        "    from vllm.entrypoints.cli.main import main\n"
        "    main()\n"
    )
    with open("/usr/local/bin/vllm_patched.py", "w") as f:
        f.write(script)
    print("Wrote /usr/local/bin/vllm_patched.py")

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("ffmpeg", "libsndfile1")
    .entrypoint([])
    .uv_pip_install(
        "vllm[audio]>=0.19.1",
        "transformers>=5.5.0",
        "requests",
        "soundfile",
        "librosa",
        "av",
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
            "VLLM_SERVER_DEV_MODE": "1",
            "NCCL_DEBUG": "OFF",
            "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        }
    )
    .run_commands(
        # 1. Patch the vLLM gpu_model_runner cache bug
        'python -c \'import sys; f="/usr/local/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"; c=open(f).read(); c=c.replace("cache_tensor.zero_()", "[t.zero_() for t in cache_tensor if t is not None] if isinstance(cache_tensor, (list, tuple)) else cache_tensor.zero_()"); open(f,"w").write(c)\'',

        # 2. Expand the audio mask to the TRUE length of the sequence
        'python -c \'import sys; f="/usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py"; c=open(f).read(); c=c.replace("hidden_states = hidden_states * mask[:, None, :, None]", "mask = torch.ones(hidden_states.shape[0], hidden_states.shape[2], dtype=torch.bool, device=hidden_states.device) if mask.dim() == 1 else mask; hidden_states = hidden_states * mask[:, None, :, None]"); open(f,"w").write(c)\'',

        # 3. Cast audio encodings to match embed_audio weight dtype
        'python -c \'import sys; f="/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/gemma4_mm.py"; c=open(f).read(); c=c.replace("audio_features = self.embed_audio(inputs_embeds=audio_encodings)", "audio_features = self.embed_audio(inputs_embeds=audio_encodings.to(next(self.embed_audio.parameters()).dtype))"); open(f,"w").write(c)\'',
    )
    # 4. Write the patched vLLM entrypoint via a Python function (no shell-escaping issues)
    .run_function(write_patched_entrypoint)
    .run_function(download_model, secrets=[modal.Secret.from_name("hf-secret")])
)

app = modal.App("example-gemma-4-e2b-Autoround-it-inference")

cache_vol = modal.Volume.from_name("vllm-compile-cache6", create_if_missing=True)

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
            # Patched entrypoint: registers Gemma4UnifiedAssistantConfig and
            # guards main() behind if __name__ == '__main__' for spawn safety.
            "python", "/usr/local/bin/vllm_patched.py",
            "serve",
            MODEL_PATH,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            "32768",
            "--kv-cache-dtype",
            "fp8",
            "--gpu-memory-utilization",
            "0.85",
            "--max-num-batched-tokens",
            "4096",
            "--block-size",
            "32",
            "--max-num-seqs",
            "8",
            "--reasoning-parser", "gemma4",
            "--enable-prefix-caching",
            "--generation-config",
            "vllm",
            "--disable-custom-all-reduce",
            "--trust-remote-code",
            "--disable-log-stats",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--enable-sleep-mode",
            "--enable-chunked-prefill",
            "--async-scheduling",
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
            "--limit-mm-per-prompt", '{"image": 0, "audio": 0}',
            "--speculative-config", '{"model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":1}',
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