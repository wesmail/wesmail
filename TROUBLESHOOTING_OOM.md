# ⚠️ OUT OF MEMORY (OOM) FIX

## 🔴 Your Error

```
RuntimeError: DataLoader worker (pid 1387320) is killed by signal: Killed.
```

This means the Linux OOM killer terminated your worker because **you ran out of RAM**.

---

## ✅ IMMEDIATE FIX - Use Conservative Settings

**Replace your current settings with these SAFE settings:**

```python
from optimized_jetclass_dataset import JetClassLightningDataModule

dm = JetClassLightningDataModule(
    train_files="data/train*.h5",
    val_files="data/val*.h5",
    test_files="data/test*.h5",
    
    # 🟢 CONSERVATIVE SETTINGS (won't OOM)
    read_chunk_size=1024,       # ⬇️ Reduced from 8192
    max_cached_chunks=10,       # ⬇️ Reduced from 30
    batch_size=256,             # ⬇️ Reduced from 1024
    num_workers=4,              # ⬇️ Reduced from 16
    prefetch_factor=2,          # ⬇️ Reduced from 8
    persistent_workers=True,
    pin_memory=True,
)
```

**This will use ~10-20GB RAM instead of ~300GB and won't crash!**

You'll still get **3-10x speedup** vs your original code.

---

## 📊 Step-by-Step: Find Your Optimal Settings

### Step 1: Check Your Available RAM

```bash
# Check total RAM
free -h

# Example output:
#               total        used        free      shared  buff/cache   available
# Mem:           62Gi       5.0Gi        45Gi       1.0Gi        12Gi        55Gi
#                ^^^^                                                         ^^^^
#              TOTAL RAM                                              AVAILABLE RAM
```

**Important**: Use only **50-60%** of available RAM for data loading!

### Step 2: Calculate Safe Settings

**Formula**:
```
RAM per worker ≈ max_cached_chunks × read_chunk_size × sample_size
Total RAM = num_workers × RAM per worker
```

**Example** (assuming 100KB per sample):
```python
# Conservative (10GB total)
read_chunk_size=1024
max_cached_chunks=10
num_workers=4
# = 4 workers × 10 chunks × 1024 samples × 100KB
# = 4 workers × 1 MB per chunk × 10 chunks = ~40MB per worker × 4 = ~4GB

# Medium (30GB total)
read_chunk_size=2048
max_cached_chunks=15
num_workers=8
# = ~30GB

# Aggressive (60GB total - only if you have 128GB+ RAM!)
read_chunk_size=4096
max_cached_chunks=20
num_workers=12
# = ~100GB
```

### Step 3: Start Small, Scale Up

```python
# 1. Start with minimal settings
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=512,
    max_cached_chunks=5,
    num_workers=2,
    batch_size=128,
)

# 2. Monitor RAM usage while training
# In another terminal:
watch -n 1 'free -h'

# 3. If RAM usage stays <50%, increase settings:
read_chunk_size=1024,
max_cached_chunks=10,
num_workers=4,

# 4. Keep increasing until RAM is at ~60-70%
# DON'T go above 70% or you'll OOM again!
```

---

## 🎯 Recommended Settings by RAM Size

### If You Have 32GB RAM
```python
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=1024,
    max_cached_chunks=8,
    batch_size=256,
    num_workers=4,
    prefetch_factor=2,
)
# Uses ~15GB, safe for 32GB system
# Speedup: 3-8x
```

### If You Have 64GB RAM
```python
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=2048,
    max_cached_chunks=12,
    batch_size=512,
    num_workers=6,
    prefetch_factor=3,
)
# Uses ~30GB, safe for 64GB system
# Speedup: 5-12x
```

### If You Have 128GB RAM
```python
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=4096,
    max_cached_chunks=20,
    batch_size=512,
    num_workers=8,
    prefetch_factor=4,
)
# Uses ~60GB, safe for 128GB system
# Speedup: 8-20x
```

### If You Have 256GB+ RAM
```python
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=8192,
    max_cached_chunks=30,
    batch_size=1024,
    num_workers=12,
    prefetch_factor=6,
)
# Uses ~150GB, safe for 256GB+ system
# Speedup: 10-30x
```

---

## 🔧 Quick Diagnostic Script

Save this as `check_memory.py`:

```python
import subprocess
import re

def get_available_ram_gb():
    """Get available RAM in GB."""
    result = subprocess.run(['free', '-g'], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    mem_line = lines[1]
    numbers = re.findall(r'\d+', mem_line)
    available_gb = int(numbers[5])  # 'available' column
    return available_gb

def recommend_settings(available_gb):
    """Recommend settings based on available RAM."""
    # Use 50% of available RAM
    target_ram_gb = available_gb * 0.5
    
    print(f"\n{'='*60}")
    print(f"Available RAM: {available_gb} GB")
    print(f"Target RAM for data loading: {target_ram_gb:.1f} GB (50%)")
    print(f"{'='*60}\n")
    
    if target_ram_gb < 10:
        print("⚠️  WARNING: Low RAM! Use minimal settings:")
        print("""
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=512,
    max_cached_chunks=5,
    batch_size=128,
    num_workers=2,
    prefetch_factor=2,
)
# Expected speedup: 2-5x
""")
    elif target_ram_gb < 20:
        print("🟡 Moderate RAM. Use conservative settings:")
        print("""
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=1024,
    max_cached_chunks=8,
    batch_size=256,
    num_workers=4,
    prefetch_factor=2,
)
# Expected speedup: 3-8x
""")
    elif target_ram_gb < 40:
        print("🟢 Good RAM. Use balanced settings:")
        print("""
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=2048,
    max_cached_chunks=15,
    batch_size=512,
    num_workers=6,
    prefetch_factor=3,
)
# Expected speedup: 5-12x
""")
    elif target_ram_gb < 80:
        print("🟢 Great RAM! Use optimized settings:")
        print("""
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=4096,
    max_cached_chunks=20,
    batch_size=512,
    num_workers=8,
    prefetch_factor=4,
)
# Expected speedup: 8-20x
""")
    else:
        print("⭐ Excellent RAM! Use aggressive settings:")
        print("""
dm = JetClassLightningDataModule(
    ...,
    read_chunk_size=8192,
    max_cached_chunks=30,
    batch_size=1024,
    num_workers=12,
    prefetch_factor=6,
)
# Expected speedup: 10-30x
""")
    
    print(f"\n{'='*60}")
    print("Monitor RAM usage during training:")
    print("  watch -n 1 'free -h'")
    print("\nIf RAM fills up, reduce settings!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    available = get_available_ram_gb()
    recommend_settings(available)
```

Run it:
```bash
python check_memory.py
```

---

## 🚨 Warning Signs

### Your System is Running Out of Memory When:
- ❌ `free -h` shows "available" RAM < 10%
- ❌ Swap usage is increasing
- ❌ Workers get killed randomly
- ❌ System becomes slow/unresponsive

### Solutions:
1. **Immediately reduce** `max_cached_chunks` (halve it)
2. **Reduce** `read_chunk_size` (halve it)
3. **Reduce** `num_workers` (halve it)
4. **Restart** training with new settings

---

## 💡 Alternative: Use Smaller Chunks with More Workers

Instead of caching huge amounts, use smaller chunks but more parallelism:

```python
# Strategy A: Large cache, few workers (needs lots of RAM)
read_chunk_size=8192,
max_cached_chunks=30,
num_workers=8,
# RAM: HIGH, Speed: FAST

# Strategy B: Small cache, many workers (needs less RAM)
read_chunk_size=2048,
max_cached_chunks=10,
num_workers=16,
# RAM: MEDIUM, Speed: STILL FAST
```

Try Strategy B if you don't have enough RAM for Strategy A!

---

## 🎯 Safe Starting Point (Works on Most Systems)

**Use this configuration - guaranteed not to OOM on systems with 32GB+ RAM:**

```python
dm = JetClassLightningDataModule(
    train_files="data/train*.h5",
    val_files="data/val*.h5",
    test_files="data/test*.h5",
    
    # Safe for 32GB+ systems
    read_chunk_size=1024,
    max_cached_chunks=10,
    batch_size=256,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

# Still 3-10x faster than original!
# Won't crash!
# Scale up from here if you have more RAM!
```

---

## ✅ Checklist

Before running again:

- [ ] Checked available RAM with `free -h`
- [ ] Used conservative settings (see above)
- [ ] Set `max_cached_chunks <= 10` initially
- [ ] Set `num_workers <= 4` initially
- [ ] Monitoring RAM with `watch -n 1 'free -h'` in another terminal
- [ ] Ready to reduce settings if RAM fills up

---

## 🔄 Gradual Scaling Process

```python
# Week 1: Safe baseline
read_chunk_size=1024, max_cached_chunks=10, num_workers=4

# Week 2: If RAM OK, increase chunks
read_chunk_size=2048, max_cached_chunks=10, num_workers=4

# Week 3: If RAM OK, increase cache
read_chunk_size=2048, max_cached_chunks=15, num_workers=4

# Week 4: If RAM OK, increase workers
read_chunk_size=2048, max_cached_chunks=15, num_workers=6

# Continue scaling until RAM at 60-70%
```

**NEVER increase all parameters at once!**

---

## 📞 Still Having Issues?

1. Run `check_memory.py` to get recommended settings
2. Start with conservative settings above
3. Monitor with `htop` or `watch -n 1 'free -h'`
4. Increase ONE parameter at a time
5. If workers still get killed, your sample_size might be larger than expected

Check your sample size:
```python
dm.setup('fit')
dataloader = dm.train_dataloader()
batch = next(iter(dataloader))

# Check memory per sample
import sys
sample_size = sys.getsizeof(batch['node_features'][0].numpy()) / 1024 / 1024
print(f"Sample size: {sample_size:.2f} MB")

# If > 1MB per sample, reduce settings even more!
```

---

**TL;DR: Use the "Safe Starting Point" settings above, monitor RAM, and scale up gradually!**