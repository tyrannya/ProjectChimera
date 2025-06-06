import time
import bentoml
from bentoml.io import NumpyNdarray, Text
import torch

model = bentoml.torchscript.load_model("nn_predictor:prod")
svc = bentoml.Service("nn_predictor_svc")

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def predict(input_array):
    start = time.monotonic()
    with torch.no_grad():
        tensor = torch.from_numpy(input_array).float()
        out = model(tensor)
    latency = (time.monotonic() - start) * 1000
    svc.ctx.logger.info(f"latency_ms={latency:.2f}")
    return out.cpu().numpy()

@svc.api(input=None, output=Text(), route="/livez")
async def livez():
    return "ok"
