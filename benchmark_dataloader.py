"""
Benchmark script to measure data loading performance.
Compare original vs optimized configurations.
"""

import time
import torch
import numpy as np
from pathlib import Path
from optimized_jetclass_dataset import JetClassLightningDataModule


def benchmark_dataloader(datamodule, num_batches=100, warmup_batches=10):
    """
    Benchmark data loading speed.
    
    Args:
        datamodule: Lightning data module
        num_batches: Number of batches to time
        warmup_batches: Number of warmup batches (excluded from timing)
    
    Returns:
        dict with timing statistics
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {num_batches} batches (after {warmup_batches} warmup)")
    print(f"{'='*60}")
    
    # Setup
    datamodule.setup('fit')
    dataloader = datamodule.train_dataloader()
    
    # Warmup
    print(f"Warming up cache...")
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break
    
    # Benchmark
    print(f"Starting benchmark...")
    batch_times = []
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Simulate minimal processing (just move to device)
        if torch.cuda.is_available():
            batch['node_features'] = batch['node_features'].cuda(non_blocking=True)
            batch['labels'] = batch['labels'].cuda(non_blocking=True)
        
        batch_times.append(time.time() - batch_start)
        
        if i >= num_batches - 1:
            break
    
    total_time = time.time() - start_time
    
    # Statistics
    batch_times = np.array(batch_times)
    stats = {
        'total_time': total_time,
        'batches': len(batch_times),
        'mean_batch_time': batch_times.mean(),
        'std_batch_time': batch_times.std(),
        'min_batch_time': batch_times.min(),
        'max_batch_time': batch_times.max(),
        'median_batch_time': np.median(batch_times),
        'throughput': len(batch_times) * dataloader.batch_size / total_time,
    }
    
    return stats


def print_stats(name, stats):
    """Pretty print benchmark statistics."""
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    print(f"Total time:       {stats['total_time']:.2f}s")
    print(f"Batches:          {stats['batches']}")
    print(f"Mean batch time:  {stats['mean_batch_time']*1000:.1f}ms ± {stats['std_batch_time']*1000:.1f}ms")
    print(f"Min batch time:   {stats['min_batch_time']*1000:.1f}ms")
    print(f"Max batch time:   {stats['max_batch_time']*1000:.1f}ms")
    print(f"Median batch time:{stats['median_batch_time']*1000:.1f}ms")
    print(f"Throughput:       {stats['throughput']:.1f} samples/sec")
    print(f"{'='*60}")


def compare_configurations(
    train_files,
    val_files,
    test_files,
    num_batches=100
):
    """
    Compare different configurations.
    """
    print("\n" + "="*60)
    print("🚀 DATA LOADING BENCHMARK")
    print("="*60)
    
    configurations = [
        {
            'name': 'BASELINE (Original Settings)',
            'params': {
                'read_chunk_size': 128,
                'max_cached_chunks': 1,
                'batch_size': 256,
                'num_workers': 4,
                'prefetch_factor': 2,
            }
        },
        {
            'name': 'OPTIMIZED (Moderate)',
            'params': {
                'read_chunk_size': 2048,
                'max_cached_chunks': 20,
                'batch_size': 512,
                'num_workers': 8,
                'prefetch_factor': 4,
            }
        },
        {
            'name': 'OPTIMIZED (Aggressive)',
            'params': {
                'read_chunk_size': 4096,
                'max_cached_chunks': 40,
                'batch_size': 1024,
                'num_workers': 12,
                'prefetch_factor': 6,
            }
        },
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n\n{'#'*60}")
        print(f"Testing: {config['name']}")
        print(f"{'#'*60}")
        print(f"Settings: {config['params']}")
        
        try:
            # Create datamodule
            dm = JetClassLightningDataModule(
                train_files=train_files,
                val_files=val_files,
                test_files=test_files,
                **config['params']
            )
            
            # Benchmark
            stats = benchmark_dataloader(dm, num_batches=num_batches)
            stats['config'] = config['name']
            results.append(stats)
            
            print_stats(config['name'], stats)
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            continue
    
    # Compare results
    if len(results) > 1:
        print("\n\n" + "="*60)
        print("📈 COMPARISON")
        print("="*60)
        
        baseline = results[0]
        print(f"\nBaseline: {baseline['config']}")
        print(f"  Throughput: {baseline['throughput']:.1f} samples/sec")
        print(f"  Avg batch:  {baseline['mean_batch_time']*1000:.1f}ms")
        
        for result in results[1:]:
            speedup = result['throughput'] / baseline['throughput']
            time_reduction = (1 - result['mean_batch_time'] / baseline['mean_batch_time']) * 100
            
            print(f"\n{result['config']}:")
            print(f"  Throughput: {result['throughput']:.1f} samples/sec")
            print(f"  Avg batch:  {result['mean_batch_time']*1000:.1f}ms")
            print(f"  ⚡ SPEEDUP:  {speedup:.2f}x faster")
            print(f"  ⏱️  TIME SAVE: {time_reduction:.1f}% reduction in batch time")
            
            # Epoch estimate
            if baseline['throughput'] > 0:
                samples_per_epoch = 10_000_000  # Example: 10M events
                baseline_epoch_time = samples_per_epoch / baseline['throughput'] / 3600
                optimized_epoch_time = samples_per_epoch / result['throughput'] / 3600
                
                print(f"\n  Estimated epoch time (10M events):")
                print(f"    Baseline:  {baseline_epoch_time:.1f} hours")
                print(f"    Optimized: {optimized_epoch_time:.1f} hours")
                print(f"    💰 SAVINGS: {baseline_epoch_time - optimized_epoch_time:.1f} hours per epoch!")
    
    return results


def quick_benchmark(train_files, num_batches=50):
    """
    Quick benchmark with single optimized configuration.
    """
    print("\n" + "="*60)
    print("🚀 QUICK BENCHMARK - Optimized Configuration")
    print("="*60)
    
    dm = JetClassLightningDataModule(
        train_files=train_files,
        val_files=train_files,  # Use same for quick test
        test_files=train_files,
        read_chunk_size=2048,
        max_cached_chunks=30,
        batch_size=512,
        num_workers=8,
        prefetch_factor=4,
        persistent_workers=True,
    )
    
    stats = benchmark_dataloader(dm, num_batches=num_batches, warmup_batches=5)
    print_stats("Optimized Configuration", stats)
    
    # Extrapolate to full epoch
    samples_per_epoch = 10_000_000  # Adjust to your dataset size
    epoch_time_hours = samples_per_epoch / stats['throughput'] / 3600
    
    print(f"\n📊 EXTRAPOLATION (for {samples_per_epoch:,} events):")
    print(f"  Epoch time: ~{epoch_time_hours:.1f} hours")
    print(f"  10 epochs:  ~{epoch_time_hours * 10:.1f} hours")
    print(f"  100 epochs: ~{epoch_time_hours * 100:.1f} hours")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark HDF5 data loading")
    parser.add_argument("--train_files", type=str, required=True,
                       help="Path to training HDF5 files (supports globs)")
    parser.add_argument("--val_files", type=str, default=None,
                       help="Path to validation HDF5 files (optional)")
    parser.add_argument("--test_files", type=str, default=None,
                       help="Path to test HDF5 files (optional)")
    parser.add_argument("--num_batches", type=int, default=100,
                       help="Number of batches to benchmark")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with single config")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple configurations")
    
    args = parser.parse_args()
    
    val_files = args.val_files or args.train_files
    test_files = args.test_files or args.train_files
    
    if args.quick:
        # Quick single-config benchmark
        quick_benchmark(args.train_files, num_batches=args.num_batches)
    elif args.compare:
        # Compare multiple configurations
        compare_configurations(
            args.train_files,
            val_files,
            test_files,
            num_batches=args.num_batches
        )
    else:
        # Default: quick benchmark
        quick_benchmark(args.train_files, num_batches=args.num_batches)