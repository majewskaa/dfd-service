import pytorch_lightning as pl
import torch
import gc

class CUDAMemoryMonitor(pl.Callback):
    """
    Callback to monitor and log CUDA memory usage during training.
    """

    def __init__(self):
        super().__init__()
        self.baseline_memory = 0
        self.max_input_size = 0

    def on_train_start(self, trainer, pl_module):
        if not torch.cuda.is_available():
            print("CUDA is not available. Memory monitoring disabled.")
            return
        
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        
        self.baseline_memory = torch.cuda.memory_allocated()
        print(f"\n[Memory] Baseline (Model + Optimizer): {self._format_bytes(self.baseline_memory)}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not torch.cuda.is_available():
            return
        
        # Calculate input batch size
        batch_size_bytes = 0
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    batch_size_bytes += v.element_size() * v.nelement()
        elif isinstance(batch, (list, tuple)):
            for v in batch:
                if isinstance(v, torch.Tensor):
                    batch_size_bytes += v.element_size() * v.nelement()
        elif isinstance(batch, torch.Tensor):
            batch_size_bytes = batch.element_size() * batch.nelement()
            
        self.max_input_size = max(self.max_input_size, batch_size_bytes)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not torch.cuda.is_available():
            return
            
        # We only log periodically to avoid spam, or we could log to wandb
        # For this task, the user wants an overview. Let's print a summary at the end of the epoch
        # or we could log peak stats here.
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        if not torch.cuda.is_available():
            return

        current_alloc = torch.cuda.memory_allocated()
        max_alloc = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        # Estimate "Data/Activation" as Max - Baseline (rough approximation)
        # This assumes baseline is mostly static model weights + optimizer states
        dynamic_peak = max_alloc - self.baseline_memory
        
        print(f"\n{'='*40}")
        print(f"CUDA Memory Overview (Epoch {trainer.current_epoch})")
        print(f"{'='*40}")
        print(f"Baseline (Model/Opt): {self._format_bytes(self.baseline_memory):>10}")
        print(f"Peak Allocated:       {self._format_bytes(max_alloc):>10}")
        print(f"Current Allocated:    {self._format_bytes(current_alloc):>10}")
        print(f"Reserved (Cached):    {self._format_bytes(reserved):>10}")
        print(f"Est. Data/Activations:{self._format_bytes(dynamic_peak):>10}")
        print(f"Input Batch Size:     {self._format_bytes(self.max_input_size):>10}")
        print(f"{'='*40}\n")
        
        # Reset peak stats for the next epoch to get fresh peak readings
        torch.cuda.reset_peak_memory_stats()
        self.max_input_size = 0

    @staticmethod
    def _format_bytes(size):
        power = 1024
        n = 0
        power_labels = {0 : '', 1: 'KiB', 2: 'MiB', 3: 'GiB', 4: 'TiB'}
        while size > power:
            size /= power
            n += 1
        return f"{size:.2f} {power_labels[n]}"
