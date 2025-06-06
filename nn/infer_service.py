import time
import traceback # Added
import os # Added
import sys # Added
import bentoml
from bentoml.io import NumpyNdarray, Text
from bentoml.exceptions import InternalServerError, ServiceUnavailable # Added ServiceUnavailable
import torch
import numpy as np

# Add project root to sys.path to allow importing tools.telegram_notifier
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from tools.telegram_notifier import TelegramNotifier
except ImportError as e:
    print(f"CRITICAL: Failed to import TelegramNotifier in nn/infer_service.py: {e}. Notifications will be disabled.")
    TelegramNotifier = None

# model is loaded by BentoML, its type is typically Union[torch.ScriptModule, torch.nn.Module]
# For simplicity, we can use a general type like 'Any' or a more specific one if known.
# Let's assume model is a torch.ScriptModule as per bentoml.torchscript.load_model
model: torch.ScriptModule = bentoml.torchscript.load_model("nn_predictor:prod")
svc: bentoml.Service = bentoml.Service("nn_predictor_svc")

notifier = None
if TelegramNotifier:
    try:
        # Attempt to load .env file if python-dotenv is available for local dev
        from dotenv import load_dotenv
        dotenv_path = os.path.join(project_root, '.env') # .env in project root
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            # svc.ctx.logger.info(f"InferService: Loaded .env from {dotenv_path}") # Requires svc to be fully init
        # else:
            # svc.ctx.logger.info(f"InferService: .env not found at {dotenv_path}")

        notifier = TelegramNotifier()
        # svc.ctx.logger.info("InferService: TelegramNotifier initialized successfully.") # svc.ctx.logger may not be ready here
        print("InferService: TelegramNotifier initialized successfully.")
    except ImportError:
        # svc.ctx.logger.warning("InferService: python-dotenv not found. Cannot load .env. Relying on env vars for TelegramNotifier.")
        print("InferService: python-dotenv not found. Cannot load .env. Relying on env vars for TelegramNotifier.")
        try:
            notifier = TelegramNotifier() # Try again, relying purely on env vars
            print("InferService: TelegramNotifier initialized successfully (no dotenv).")
        except ValueError as ve_no_dotenv:
            print(f"InferService: Failed to initialize TelegramNotifier (dotenv not found, env vars likely missing): {ve_no_dotenv}")

    except ValueError as e:
        # svc.ctx.logger.warning(f"InferService: Failed to initialize TelegramNotifier: {e}. Service will run without Telegram notifications.")
        print(f"InferService: Failed to initialize TelegramNotifier: {e}. Service will run without Telegram notifications.")
    except Exception as e:
        # svc.ctx.logger.error(f"InferService: Unexpected error during TelegramNotifier initialization: {e}")
        print(f"InferService: Unexpected error during TelegramNotifier initialization: {e}")
else:
    print("InferService: TelegramNotifier class not available. Notifications disabled.")


@svc.api(input=NumpyNdarray(), output=NumpyNdarray()) # Input is np.ndarray, output is np.ndarray
async def predict(input_array: np.ndarray) -> np.ndarray:
    start_time: float = time.monotonic()
    try:
        with torch.no_grad():
            tensor: torch.Tensor = torch.from_numpy(input_array).float()
            # Assuming the model might be on GPU, ensure tensor is moved to the model's device
            # This is a good practice, though load_model might handle it for some runners.
            # If model_device is known (e.g., "cuda" or "cpu"), use it.
            # For now, let's assume model and tensor are on compatible devices or model handles it.
            out: torch.Tensor = model(tensor)

        output_numpy: np.ndarray = out.cpu().numpy()
        latency_ms: float = (time.monotonic() - start_time) * 1000
        svc.ctx.logger.info(f"Prediction successful. Latency: {latency_ms:.2f}ms")
        return output_numpy

    except Exception as e:
        svc.ctx.logger.error(
            f"Prediction error: {e}", exc_info=True
        )
        # Calculate latency even in case of error, if meaningful
        latency_ms_error: float = (time.monotonic() - start_time) * 1000
        svc.ctx.logger.info(
            f"Prediction failed. Latency until error: {latency_ms_error:.2f}ms"
        )
        if notifier:
            detailed_error = traceback.format_exc()
            notifier.send_error(f"Критический сбой в nn/infer_service.py (predict endpoint):\n{detailed_error}")
        raise InternalServerError(f"Prediction failed: {str(e)}")


@svc.api(input=None, output=Text(), route="/livez") # Output is str
async def livez() -> str:
    return "ok"

@svc.api(input=None, output=Text(), route="/readyz")
async def readyz() -> str:
    # 1. Model Loaded Check
    # The model is loaded globally when the script starts.
    # If `bentoml.torchscript.load_model` failed, it would raise an error at startup.
    # So, if the service is running, the model object should exist.
    # We can add an explicit check for None, though it's unlikely to be None if service started.
    if model is None:
        error_msg = "Readiness check failed: Model is not loaded (global model object is None)."
        svc.ctx.logger.error(error_msg)
        if notifier:
            notifier.send_error(f"Критический сбой в nn/infer_service.py (readyz - model not loaded):\n{error_msg}")
        raise ServiceUnavailable("Model not loaded")

    # Check if the model is a torch.ScriptModule as expected (optional, but good for sanity)
    if not isinstance(model, torch.jit.ScriptModule):
        error_msg = f"Readiness check failed: Model is not a torch.jit.ScriptModule, type is {type(model)}."
        svc.ctx.logger.error(error_msg)
        if notifier:
            notifier.send_error(f"Критический сбой в nn/infer_service.py (readyz - model type mismatch):\n{error_msg}")
        raise ServiceUnavailable("Model is of unexpected type")

    # 2. Model Inference Sanity Check
    try:
        # Determine device for the dummy tensor based on model's device if possible,
        # or default to CPU/CUDA.
        # A simple way: check if any parameters are on CUDA.
        device_str = "cpu"
        if next(model.parameters(), None) is not None and next(model.parameters()).is_cuda:
            device_str = "cuda"

        # If model has no parameters, or to be more robust, try to use model's device directly
        # This part can be tricky if model is not yet moved to a device or device is unknown.
        # For now, let's assume CPU or check CUDA availability.
        if torch.cuda.is_available():
             # Check if model is on CUDA. If model.device exists, use it.
             # This is a heuristic; a more robust way is needed if model device is dynamic.
             try:
                 if next(model.parameters()).is_cuda:
                     device_str = "cuda"
             except StopIteration: # Model has no parameters
                 pass # Keep device_str as "cpu" or make a guess

        device = torch.device(device_str)

        # Dimensions: batch_size=1, seq_len=100, num_features=5
        # Based on nn/train.py: X.append(df.iloc[i - window:i][['close', 'return_1', 'ema_9', 'ema_21', 'volume']].values)
        # So, num_features = 5. seq_len (window) = 100.
        dummy_input = torch.randn(1, 100, 5, device=device).float()

        with torch.no_grad():
            model(dummy_input) # Perform a forward pass
        svc.ctx.logger.info("Readiness check: Model inference sanity check passed.")

    except Exception as e:
        svc.ctx.logger.error(f"Readiness check failed: Model inference sanity check error: {e}", exc_info=True)
        if notifier:
            detailed_error = traceback.format_exc()
            notifier.send_error(f"Критический сбой в nn/infer_service.py (readyz - inference check):\n{detailed_error}")
        raise ServiceUnavailable(f"Model inference check failed: {str(e)}")

    return "ok"
