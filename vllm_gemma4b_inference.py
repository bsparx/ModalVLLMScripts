import json
import socket
import subprocess
import time
from typing import Any

import aiohttp
import modal

# -------------------------------------------------------------------------
# 1. Configuration & Model Download
# -------------------------------------------------------------------------
MODEL_NAME = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
MODEL_PATH = "/model"

def download_model():
    """Bake the model weights into the container image."""
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_PATH,
        ignore_patterns=["*.pt", "*.bin"],  # Use safetensors if available
    )

# -------------------------------------------------------------------------
# 2. Image Environment & Dependencies
# -------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
       .uv_pip_install(
        "vllm==0.19.0",
    )
    .uv_pip_install(
        "transformers==5.5.0",  # Required for Gemma 4
        "requests",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "TRITON_CACHE_DIR": "/cache/triton",
            
            # Disable NCCL heartbeat monitor
            "TORCH_NCCL_ENABLE_MONITORING": "0",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
            
            # Use spawn to avoid inheriting stale sockets
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            # Single GPU – disable peer-to-peer
            "NCCL_P2P_DISABLE": "1",
            # Required for sleep/wake endpoints for snapshotting
            "VLLM_SERVER_DEV_MODE": "1",
            # Reduce NCCL verbosity
            "NCCL_DEBUG": "OFF",
            
            # Force internal vLLM/PyTorch Distributed sockets to use localhost.
            # Prevents broken pipes when memory snapshots are restored on containers with new IPs.
            "VLLM_HOST_IP": "127.0.0.1",
        }
    )
    .run_function(download_model, secrets=[modal.Secret.from_name("hf-secret")])
)

app = modal.App("example-gemma-4-vllm-snapshot-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000

# -------------------------------------------------------------------------
# 3. vLLM Server Class with Snapshot Lifecycle Hooks
# -------------------------------------------------------------------------
@app.cls(
    image=vllm_image,
    gpu=f"L4", # Since it is 4-bit, you could potentially drop this to an L4 or A100 depending on the active parameters.
    scaledown_window=15 * MINUTES,  # Stay up for 15 minutes of inactivity before sleeping
    timeout=40 * MINUTES,
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=100)
class VllmServer:
    def wait_ready(self):
        """Wait until the vLLM server is responsive on localhost."""
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
        """Start the server natively, run warmup, and trigger sleep mode for snapshot."""
        import requests

        cmd =[
            "vllm",
            "serve",
            MODEL_PATH,  # Serve the locally baked model
            "--served-model-name", MODEL_NAME, "llm", # Register "llm" as an alias
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--tensor-parallel-size", str(N_GPU),
            "--uvicorn-log-level=info",
            "--async-scheduling",
            
            # Gemma-specific configuration
            # ENABLE AUDIO INPUT (1 audio limit per prompt)
            "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0, "audio": 1}),
            "--enable-auto-tool-choice",
            "--reasoning-parser", "gemma4",
            "--tool-call-parser", "gemma4",
            "--default-chat-template-kwargs", '{"enable_thinking": false}',

            # Options required for snapshotting
            "--enable-sleep-mode",
            "--disable-log-stats",
        ]

        print("Starting vLLM server...")
        self.process = subprocess.Popen(cmd)

        self.wait_ready()
        print("vLLM is up! Warming up to trigger CUDA graph compilation...")

        warmup_payload = {
            "model": "llm",
            "messages":[{"role": "user", "content": "Explain the singular value decomposition."}],
            "max_tokens": 5,
            "chat_template_kwargs": {"enable_thinking": True}
        }
        try:
            requests.post(
                f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
                json=warmup_payload,
                timeout=300,
            ).raise_for_status()
            print("Warmup successful.")
        except Exception as e:
            print(f"Warmup failed: {e}")

        print("Putting vLLM to sleep for snapshotting...")
        requests.post(f"http://127.0.0.1:{VLLM_PORT}/sleep?level=1").raise_for_status()

    @modal.enter(snap=False)
    def wake_up(self):
        """Restore process from snapshot instantly on cold start."""
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


# -------------------------------------------------------------------------
# 4. Local Test Entrypoint
# -------------------------------------------------------------------------
@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None, twice=True):
    url = await VllmServer.serve.get_web_url.aio()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

    messages =[  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages)
        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}
    # explicitly enable thinking for this model
    payload["chat_template_kwargs"] = {"enable_thinking": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            # extract new content and stream it
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):  # SSE prefix
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert (
                chunk["object"] == "chat.completion.chunk"
            )  # or something went horribly wrong
            delta = chunk["choices"][0]["delta"]
            content = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
            )
            if content:
                print(content, end="")
            else:
                print("\n", chunk)
    print()