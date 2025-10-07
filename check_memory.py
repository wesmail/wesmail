#!/usr/bin/env python3
"""
Memory checker and configuration recommender for HDF5 data loading.
Run this BEFORE training to get safe settings for your system.
"""

import subprocess
import re
import sys


def get_available_ram_gb():
    """Get available RAM in GB."""
    try:
        result = subprocess.run(['free', '-g'], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        mem_line = lines[1]
        numbers = re.findall(r'\d+', mem_line)
        
        if len(numbers) >= 6:
            total_gb = int(numbers[0])
            available_gb = int(numbers[5])  # 'available' column
            return total_gb, available_gb
        else:
            print("⚠️  Could not parse memory info")
            return None, None
    except Exception as e:
        print(f"⚠️  Error checking memory: {e}")
        print("Manually check with: free -h")
        return None, None


def recommend_settings(total_gb, available_gb):
    """Recommend settings based on available RAM."""
    if total_gb is None or available_gb is None:
        print("\n⚠️  Could not determine RAM. Use conservative settings:")
        print_conservative_settings()
        return
    
    # Use 40-50% of available RAM (conservative)
    target_ram_gb = available_gb * 0.4
    
    print(f"\n{'='*70}")
    print(f"💾 MEMORY ANALYSIS")
    print(f"{'='*70}")
    print(f"Total RAM:     {total_gb} GB")
    print(f"Available RAM: {available_gb} GB")
    print(f"Target for data loading: {target_ram_gb:.1f} GB (40% of available)")
    print(f"{'='*70}\n")
    
    if available_gb < 10:
        print("🔴 CRITICAL: Very low available RAM!")
        print("   Close other applications before training.\n")
    
    # Recommend based on target RAM
    if target_ram_gb < 8:
        print("⚠️  LOW RAM - Use MINIMAL settings:\n")
        print("="*70)
        print("""
dm = JetClassLightningDataModule(
    train_files="...",
    val_files="...",
    test_files="...",
    
    # MINIMAL SETTINGS (for <20GB total RAM)
    read_chunk_size=512,
    max_cached_chunks=5,
    batch_size=128,
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

# Expected RAM usage: ~5 GB
# Expected speedup: 2-5x vs original
""")
    
    elif target_ram_gb < 15:
        print("🟡 MODERATE RAM - Use CONSERVATIVE settings:\n")
        print("="*70)
        print("""
dm = JetClassLightningDataModule(
    train_files="...",
    val_files="...",
    test_files="...",
    
    # CONSERVATIVE SETTINGS (for 32-48GB total RAM)
    read_chunk_size=1024,
    max_cached_chunks=8,
    batch_size=256,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

# Expected RAM usage: ~10-15 GB
# Expected speedup: 3-8x vs original
""")
    
    elif target_ram_gb < 30:
        print("🟢 GOOD RAM - Use BALANCED settings:\n")
        print("="*70)
        print("""
dm = JetClassLightningDataModule(
    train_files="...",
    val_files="...",
    test_files="...",
    
    # BALANCED SETTINGS (for 64-96GB total RAM)
    read_chunk_size=2048,
    max_cached_chunks=15,
    batch_size=512,
    num_workers=6,
    prefetch_factor=3,
    persistent_workers=True,
    pin_memory=True,
)

# Expected RAM usage: ~25-35 GB
# Expected speedup: 5-15x vs original
""")
    
    elif target_ram_gb < 60:
        print("🟢 GREAT RAM - Use OPTIMIZED settings:\n")
        print("="*70)
        print("""
dm = JetClassLightningDataModule(
    train_files="...",
    val_files="...",
    test_files="...",
    
    # OPTIMIZED SETTINGS (for 128-192GB total RAM)
    read_chunk_size=4096,
    max_cached_chunks=20,
    batch_size=512,
    num_workers=8,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
)

# Expected RAM usage: ~50-70 GB
# Expected speedup: 8-20x vs original
""")
    
    else:
        print("⭐ EXCELLENT RAM - Use AGGRESSIVE settings:\n")
        print("="*70)
        print("""
dm = JetClassLightningDataModule(
    train_files="...",
    val_files="...",
    test_files="...",
    
    # AGGRESSIVE SETTINGS (for 256GB+ total RAM)
    read_chunk_size=8192,
    max_cached_chunks=30,
    batch_size=1024,
    num_workers=12,
    prefetch_factor=6,
    persistent_workers=True,
    pin_memory=True,
)

# Expected RAM usage: ~100-150 GB
# Expected speedup: 10-30x vs original
""")
    
    print("="*70)
    print()
    print_monitoring_instructions()


def print_conservative_settings():
    """Print conservative settings when RAM detection fails."""
    print("""
# SAFE CONSERVATIVE SETTINGS (works on most systems with 32GB+ RAM)
dm = JetClassLightningDataModule(
    train_files="...",
    val_files="...",
    test_files="...",
    
    read_chunk_size=1024,
    max_cached_chunks=8,
    batch_size=256,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)
""")


def print_monitoring_instructions():
    """Print instructions for monitoring RAM."""
    print("📊 MONITORING INSTRUCTIONS")
    print("="*70)
    print()
    print("1. Monitor RAM during training:")
    print("   In a separate terminal, run:")
    print("   \033[1m   watch -n 1 'free -h'\033[0m")
    print()
    print("2. Watch for these warning signs:")
    print("   ❌ 'available' RAM drops below 10%")
    print("   ❌ 'swap' usage increases")
    print("   ❌ Workers get killed with 'signal: Killed'")
    print()
    print("3. If you see warnings:")
    print("   • Reduce max_cached_chunks (halve it)")
    print("   • Reduce read_chunk_size (halve it)")
    print("   • Reduce num_workers (halve it)")
    print("   • Restart training")
    print()
    print("4. If RAM stays below 60% during training:")
    print("   • You can gradually increase settings")
    print("   • Increase ONE parameter at a time")
    print("   • Monitor after each change")
    print()
    print("="*70)
    print()
    print("✅ START CONSERVATIVE, SCALE UP GRADUALLY!")
    print()


def check_gpu_memory():
    """Check GPU memory if available."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, 
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        print(f"\n{'='*70}")
        print(f"🎮 GPU MEMORY")
        print(f"{'='*70}")
        
        for i, line in enumerate(lines):
            total, free = map(int, line.split(','))
            total_gb = total / 1024
            free_gb = free / 1024
            print(f"GPU {i}: {total_gb:.1f} GB total, {free_gb:.1f} GB free")
        
        print()
        print("💡 Batch size recommendations based on GPU memory:")
        if total_gb < 8:
            print(f"   batch_size=128-256 (for {total_gb:.0f}GB GPU)")
        elif total_gb < 16:
            print(f"   batch_size=256-512 (for {total_gb:.0f}GB GPU)")
        elif total_gb < 32:
            print(f"   batch_size=512-1024 (for {total_gb:.0f}GB GPU)")
        else:
            print(f"   batch_size=1024-2048 (for {total_gb:.0f}GB GPU)")
        
        print(f"{'='*70}\n")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # No GPU or nvidia-smi not available
        pass


def main():
    """Main function."""
    print("\n" + "="*70)
    print("🚀 HDF5 DataLoader Configuration Recommender")
    print("="*70)
    print()
    print("Analyzing your system to recommend optimal settings...")
    print()
    
    # Check RAM
    total_gb, available_gb = get_available_ram_gb()
    recommend_settings(total_gb, available_gb)
    
    # Check GPU
    check_gpu_memory()
    
    print("💾 To save these recommendations:")
    print("   python check_memory.py > my_config.txt")
    print()


if __name__ == "__main__":
    main()