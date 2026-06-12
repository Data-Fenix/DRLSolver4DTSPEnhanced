import sys, torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())

try:
    import torch_directml as dml
    print("torch_directml version:", getattr(dml, "__version__", "unknown"))

    # Try device strings first (works when backend is registered)
    for candidate in ("dml", "privateuseone:0"):
        try:
            dev = torch.device(candidate)
            x = torch.randn(2,2, device=dev)
            print(f"DirectML OK via torch.device('{candidate}'): {x.device}")
            raise SystemExit(0)
        except Exception:
            pass

    # Try module attribute in both forms (callable or prebuilt device)
    dev_attr = getattr(dml, "device", None)
    if dev_attr is None:
        raise RuntimeError("torch_directml.device not found")

    if callable(dev_attr):
        dev = dev_attr()
    else:
        dev = dev_attr  # prebuilt torch.device

    x = torch.randn(2,2, device=dev)
    print("DirectML OK via torch_directml.device:", x.device)

except Exception as e:
    print("DirectML not available:", repr(e))