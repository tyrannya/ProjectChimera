import bentoml
from bentoml.io import NumpyNdarray

model = bentoml.torchscript.load_model("nn_predictor:prod")
svc = bentoml.Service("nn_predictor_svc")

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(x):
    return model(x).cpu().numpy()
