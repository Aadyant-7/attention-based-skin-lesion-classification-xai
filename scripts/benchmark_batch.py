"""Choose a conservative training batch using measured peak allocated VRAM."""
import time
import torch
from src.models import EfficientNetCBAM


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    for batch in (16, 32, 64):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = EfficientNetCBAM(pretrained=False).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            x = torch.randn(batch, 3, 224, 224, device=device)
            y = torch.randint(0, 7, (batch,), device=device)
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**20
            print(f"batch={batch} step_seconds={time.monotonic()-start:.2f} peak_allocated_mib={peak:.0f} total_mib={total/2**20:.0f}", flush=True)
            del model, optimizer, scaler, x, y, loss
        except torch.cuda.OutOfMemoryError:
            print(f"batch={batch} OOM", flush=True)
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
