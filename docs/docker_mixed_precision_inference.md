# Docker mixed-precision inference

This service loads one configured model during FastAPI startup through
`BBoxGroundingInferenceEngine.from_pipeline_config(config)`. It does not convert
weights to GGUF, TensorRT, AWQ, or another format; the model, adapter, processor,
and config are read-only mounted files.

## Build

```bash
docker build -f docker/Dockerfile.mixed-precision -t switch-kd-mixed-precision:latest .
```

## Run

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e VLM_CONFIG_PATH=/config/a4.yaml \
  -v /host/config:/config:ro -v /host/models:/models:ro \
  -v /host/adapters:/adapters:ro -v /host/deployment:/deployment:ro \
  switch-kd-mixed-precision:latest
```

Replace every `/host/...` placeholder with a host directory. The equivalent
Compose file is `docker-compose.mixed-precision.yaml`.

```bash
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/infer \
  -F "image=@example.png" \
  -F "query=List all visible interactive UI elements on this screen."
```

The host needs an NVIDIA driver, Docker, NVIDIA Container Toolkit, and a driver
that supports the CUDA runtime in the image. The driver is not included in the
image.

The service uses the existing image resize, prompt formatter, chat template,
deterministic generation, and parsing code. Docker does not re-quantize the
model. Do not change those settings for a parity run. Run one worker only:
each worker loads another copy of the model and may exhaust GPU memory. Before
production delivery, run regression on the target GPU; CUDA kernels and driver
versions can affect runtime behavior.
