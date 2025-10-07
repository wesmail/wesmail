# HDF5 Data Loading Optimization Guide

## 🚀 Major Performance Improvements

Your original code could take **100+ hours per epoch** for 10M events. This optimized version should achieve **10-50x speedup** with proper tuning.

---

## 🎯 Key Changes Made

### 1. **LRU Cache with Multiple Chunks** (Biggest Impact!)
**Problem**: Original code cached only 1 chunk per worker → poor cache hit rate with shuffling  
**Solution**: LRU cache storing 20-30 chunks → ~80-95% cache hit rate

```python
# OLD: Single chunk cache
self._cache = {}

# NEW: Multi-chunk LRU cache
self._cache = LRUCache(maxsize=20)  # Keeps 20 chunks in memory
```

### 2. **Larger Chunk Sizes**
**Problem**: 128 samples/chunk → too many HDF5 reads (HDF5 I/O overhead dominates)  
**Solution**: 1024-2048 samples/chunk → 8-16x fewer HDF5 reads

```python
# OLD
read_chunk_size=128

# NEW
read_chunk_size=2048  # Tune based on your RAM
```

### 3. **Batch Adjacency Reconstruction**
**Problem**: Reconstructing adjacency matrix per sample is slow  
**Solution**: Reconstruct entire chunk at once, cache the result

```python
# Reconstruct all adjacency matrices in chunk at once
adj_chunk = self._reconstruct_adjacency_matrix_batch(flat_adj_chunk)
# Cache the pre-reconstructed tensors
```

### 4. **Pre-converted Torch Tensors**
**Problem**: Repeated numpy→torch conversions in hot path  
**Solution**: Convert once when loading chunk, cache torch tensors

### 5. **Optional RAM Preloading**
**Problem**: HDF5 I/O is inherently slower than RAM  
**Solution**: For datasets <50GB, load entirely to RAM

```python
# If your dataset fits in RAM (~30-50GB or less)
preload_to_ram=True  # 100-1000x faster I/O!
```

### 6. **Improved HDF5 Settings**
**Problem**: Default HDF5 chunk cache too small  
**Solution**: Larger chunk cache per file handle

```python
h5py.File(path, 'r', 
    swmr=True,
    rdcc_nbytes=1024**3,  # 1GB chunk cache
    rdcc_nslots=10007     # Prime number for hash
)
```

---

## ⚙️ Recommended Settings by Dataset Size

### **Small Dataset (<10GB, <2M events)**
```python
datamodule = JetClassLightningDataModule(
    train_files="train*.h5",
    val_files="val*.h5",
    test_files="test*.h5",
    
    # PRELOAD TO RAM - Fastest option!
    preload_to_ram=True,
    
    # Large batches
    batch_size=1024,
    
    # Moderate workers
    num_workers=4,
    
    # Standard settings
    read_chunk_size=2048,
    max_cached_chunks=10,
    prefetch_factor=4,
)
```
**Expected speedup**: 50-100x (RAM vs HDF5)

---

### **Medium Dataset (10-100GB, 2-10M events)** ⭐ Most Common
```python
datamodule = JetClassLightningDataModule(
    train_files="train*.h5",
    val_files="val*.h5", 
    test_files="test*.h5",
    
    # NO RAM preload (won't fit)
    preload_to_ram=False,
    
    # LARGE CHUNKS - Critical!
    read_chunk_size=4096,  # 4K samples per chunk
    
    # MANY CACHED CHUNKS
    max_cached_chunks=50,  # 50 chunks × 4096 = 200K samples cached
    
    # Large batches
    batch_size=512,
    
    # Many workers
    num_workers=12,  # Match your CPU cores
    
    # Aggressive prefetching
    prefetch_factor=6,
    
    # Keep workers alive
    persistent_workers=True,
)
```
**Expected speedup**: 10-30x vs original

**Memory usage**: ~50 chunks × 4096 samples × 12 workers × sample_size  
For 100KB/sample: ~50 × 4096 × 12 × 100KB ≈ **240GB** (across all workers)

---

### **Large Dataset (>100GB, >10M events)** 🔥 Your Case
```python
datamodule = JetClassLightningDataModule(
    train_files="train*.h5",
    val_files="val*.h5",
    test_files="test*.h5",
    
    # NO RAM preload
    preload_to_ram=False,
    
    # HUGE CHUNKS
    read_chunk_size=8192,  # 8K samples per chunk
    
    # BALANCED CACHE (manage memory)
    max_cached_chunks=30,  # Don't exceed RAM
    
    # Large batches (maximize GPU utilization)
    batch_size=1024,  # Or higher if GPU memory allows
    
    # MANY workers (saturate I/O)
    num_workers=16,  # Or more if you have CPU cores
    
    # Aggressive prefetch
    prefetch_factor=8,
    
    persistent_workers=True,
    pin_memory=True,
)
```
**Expected speedup**: 5-20x vs original

**Memory usage**: ~30 chunks × 8192 samples × 16 workers × sample_size  
Monitor with `htop` and adjust `max_cached_chunks` if RAM fills up.

---

## 🔧 Tuning Parameters

### **read_chunk_size** (Samples per HDF5 read)
- **Smaller (512-1024)**: Less RAM, more I/O overhead
- **Larger (4096-8192)**: More RAM, fewer I/O calls ✅ Usually better
- **Rule**: Make chunk_size >> batch_size for good cache hits

### **max_cached_chunks** (LRU cache size)
- **Formula**: `max_cached_chunks = available_RAM / (read_chunk_size × sample_size × num_workers)`
- **Monitor**: Use `htop` to watch RAM usage
- **Safe start**: 20-30 chunks per worker

### **num_workers** (Parallel data loaders)
- **Rule**: Match CPU core count, or 1.5-2× cores
- **Too few**: GPU idles waiting for data
- **Too many**: Context switching overhead
- **Profile**: Check GPU utilization with `nvidia-smi dmon`

### **batch_size** (Samples per training step)
- **Larger**: Better GPU utilization, better cache hits
- **Limit**: GPU memory
- **Recommendation**: Largest that fits in GPU memory

### **prefetch_factor** (Batches loaded ahead)
- **Larger**: Hide I/O latency better
- **Limit**: RAM usage
- **Recommendation**: 4-8 for HDF5 workloads

---

## 🎬 Quick Start Example

```python
from optimized_jetclass_dataset import JetClassLightningDataModule
import lightning.pytorch as pl

# 1. Create data module with optimized settings
dm = JetClassLightningDataModule(
    train_files="/path/to/train*.h5",
    val_files="/path/to/val*.h5",
    test_files="/path/to/test*.h5",
    
    # CRITICAL SETTINGS
    read_chunk_size=4096,      # ⬆️ Increase from 256
    max_cached_chunks=40,      # ⬆️ Increase from 1 (implicit)
    batch_size=512,            # ⬆️ Increase from 256
    num_workers=16,            # ⬆️ Increase workers
    prefetch_factor=6,         # ⬆️ Increase from 2
    
    # Optional: If dataset fits in RAM
    # preload_to_ram=True,     # 🚀 Massive speedup!
)

# 2. Setup datasets
dm.setup('fit')

# 3. Train
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='gpu',
    devices=1,
)
trainer.fit(model, dm)
```

---

## 📊 Performance Monitoring

### **Check GPU Utilization**
```bash
# Should be 90-100% during training
nvidia-smi dmon -s u
```
If GPU is <80%, data loading is bottleneck → increase `num_workers` or `prefetch_factor`

### **Check RAM Usage**
```bash
htop
```
If RAM is full → decrease `max_cached_chunks` or `read_chunk_size`

### **Profile Data Loading**
```python
import time

# Time a single epoch
dataloader = dm.train_dataloader()
start = time.time()
for batch in dataloader:
    pass  # No model forward pass
elapsed = time.time() - start
print(f"Data loading time: {elapsed:.2f}s for {len(dataloader)} batches")
```

### **Cache Hit Rate** (Add to Dataset)
```python
# In __getitem__, track hits/misses
if cached_data is None:
    self.cache_misses += 1
else:
    self.cache_hits += 1

# Print periodically
hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
print(f"Cache hit rate: {hit_rate:.2%}")  # Target: >80%
```

---

## 🚨 Common Issues

### **Out of Memory (OOM)**
**Symptom**: Process killed, RAM fills up  
**Solutions**:
1. Reduce `max_cached_chunks` (e.g., 40 → 20)
2. Reduce `read_chunk_size` (e.g., 4096 → 2048)
3. Reduce `num_workers` (e.g., 16 → 8)
4. Monitor with: `watch -n 1 free -h`

### **Slow First Epoch**
**Symptom**: First epoch is slow, subsequent epochs are fast  
**Cause**: Cache is warming up  
**Solution**: This is normal with `persistent_workers=True`

### **GPU Underutilized (<50%)**
**Symptom**: `nvidia-smi` shows low GPU usage  
**Solutions**:
1. Increase `num_workers` (e.g., 8 → 16)
2. Increase `prefetch_factor` (e.g., 2 → 6)
3. Increase `batch_size` (if GPU memory allows)

### **Still Slow?**
Check these:
1. **HDF5 file chunking**: Ensure HDF5 files use optimal chunk sizes
   ```python
   # When creating HDF5 files, use:
   f.create_dataset('feature_matrix', data=data, chunks=(1000, num_features))
   ```
2. **Disk I/O**: Use SSD, not HDD (5-10x faster)
3. **Network storage**: Avoid NFS/network drives if possible
4. **Adjacency reconstruction**: Consider `lazy_adjacency=True` if model doesn't need it immediately

---

## 💡 Advanced Optimizations

### **1. Disable Adjacency Reconstruction** (If Possible)
If your model can work with flat adjacency matrices:
```python
reconstruct_full_adjacency=False  # Skip expensive reconstruction
```
**Speedup**: 2-5x for adjacency processing

### **2. Lazy Adjacency Reconstruction**
Delay reconstruction until model actually needs it:
```python
lazy_adjacency=True  # Reconstruct on first access
```

### **3. Mixed Precision**
Not in dataset, but in training:
```python
trainer = pl.Trainer(precision='16-mixed')  # 2x faster training
```

### **4. Profile with PyTorch Profiler**
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

### **5. Preprocess HDF5 Files**
Create pre-reconstructed adjacency matrices offline:
```python
# Preprocessing script (run once)
with h5py.File('input.h5', 'r') as f_in:
    with h5py.File('output_preprocessed.h5', 'w') as f_out:
        # Copy node features
        f_out.create_dataset('feature_matrix', data=f_in['feature_matrix'][:])
        
        # Reconstruct and save adjacency matrices
        flat_adj = f_in['adjacency_matrix'][:]
        reconstructed = reconstruct_all(flat_adj)  # Batch process
        f_out.create_dataset('adjacency_matrix', data=reconstructed,
                            chunks=(100, max_particles, max_particles, n_features))
```
Then set `reconstruct_full_adjacency=False` → 5-10x speedup

---

## 📈 Expected Performance

| Dataset Size | Original | Optimized (no RAM) | Optimized (RAM) |
|--------------|----------|-------------------|-----------------|
| 1M events    | ~10 hrs  | ~30 min           | ~5 min          |
| 10M events   | ~100 hrs | ~5 hrs            | ~30 min         |
| 50M events   | ~500 hrs | ~25 hrs           | N/A (too large) |

*Assumes: 16 workers, chunk_size=4096, cached_chunks=40, batch=512*

---

## ✅ Checklist for Maximum Performance

- [ ] Use `read_chunk_size >= 2048` (4096-8192 for large datasets)
- [ ] Set `max_cached_chunks >= 20` (30-50 for large datasets)
- [ ] Set `num_workers` to CPU core count or higher (12-16)
- [ ] Set `prefetch_factor >= 4` (6-8 for HDF5)
- [ ] Use `batch_size >= 512` (largest that fits in GPU)
- [ ] Enable `persistent_workers=True`
- [ ] Enable `pin_memory=True` for GPU training
- [ ] Use SSD storage (not HDD)
- [ ] Monitor GPU utilization (should be >90%)
- [ ] Monitor RAM usage (should not swap)
- [ ] Consider `preload_to_ram=True` for small datasets
- [ ] Profile and iterate!

---

## 🆘 Need Help?

1. **Check GPU utilization**: `nvidia-smi dmon`
2. **Check RAM usage**: `htop`
3. **Profile data loading**: See "Performance Monitoring" section
4. **Start conservative**: Begin with smaller settings and scale up
5. **Measure incrementally**: Change one parameter at a time

Good luck! With these optimizations, you should see **10-50x speedup** depending on your configuration. 🚀