#!/usr/bin/env python3
"""
Lambda Labs VM Resource Monitor for Critical Weight Analysis
Monitors GPU, CPU, memory, and disk usage during experiments.
"""

import time
import psutil
import GPUtil
from datetime import datetime
import json
import os

def get_system_info():
    """Get comprehensive system information."""
    
    # GPU Info
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            'id': gpu.id,
            'name': gpu.name,
            'memory_total': f"{gpu.memoryTotal}MB",
            'memory_used': f"{gpu.memoryUsed}MB", 
            'memory_free': f"{gpu.memoryFree}MB",
            'memory_util': f"{gpu.memoryUtil:.1%}",
            'gpu_util': f"{gpu.load:.1%}",
            'temperature': f"{gpu.temperature}°C"
        })
    
    # CPU & Memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Cache directories
    cache_info = {}
    cache_dirs = [
        '/data/cache/hf',
        '/data/cache/pip', 
        '/data/cache/torch'
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                cache_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(cache_dir)
                    for filename in filenames
                )
                cache_info[cache_dir] = f"{cache_size / 1024**3:.2f}GB"
            except:
                cache_info[cache_dir] = "Error reading"
        else:
            cache_info[cache_dir] = "Not found"
    
    return {
        'timestamp': datetime.now().isoformat(),
        'gpus': gpu_info,
        'cpu_percent': f"{cpu_percent:.1f}%",
        'memory': {
            'total': f"{memory.total / 1024**3:.1f}GB",
            'used': f"{memory.used / 1024**3:.1f}GB",
            'available': f"{memory.available / 1024**3:.1f}GB",
            'percent': f"{memory.percent:.1f}%"
        },
        'disk': {
            'total': f"{disk.total / 1024**3:.1f}GB",
            'used': f"{disk.used / 1024**3:.1f}GB", 
            'free': f"{disk.free / 1024**3:.1f}GB",
            'percent': f"{(disk.used/disk.total)*100:.1f}%"
        },
        'cache_usage': cache_info
    }

def print_system_info(info):
    """Print system information in a nice format."""
    
    print(f"\n{'='*60}")
    print(f"Lambda Labs VM Monitor - {info['timestamp']}")
    print(f"{'='*60}")
    
    # GPU Information
    print(f"\n🔥 GPU Information:")
    for gpu in info['gpus']:
        print(f"  GPU {gpu['id']}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_used']}/{gpu['memory_total']} ({gpu['memory_util']})")
        print(f"    Utilization: {gpu['gpu_util']} | Temperature: {gpu['temperature']}")
    
    # System Resources
    print(f"\n💻 System Resources:")
    print(f"  CPU Usage: {info['cpu_percent']}")
    print(f"  Memory: {info['memory']['used']}/{info['memory']['total']} ({info['memory']['percent']})")
    print(f"  Disk: {info['disk']['used']}/{info['disk']['total']} ({info['disk']['percent']})")
    
    # Cache Usage
    print(f"\n📦 Cache Usage:")
    for cache_dir, size in info['cache_usage'].items():
        print(f"  {cache_dir}: {size}")

def monitor_continuous(interval=30):
    """Monitor system resources continuously."""
    print("Starting continuous monitoring (Ctrl+C to stop)")
    print(f"Update interval: {interval} seconds")
    
    try:
        while True:
            info = get_system_info()
            print_system_info(info)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def save_snapshot(output_file="vm_snapshot.json"):
    """Save a snapshot of current system state."""
    info = get_system_info()
    
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"System snapshot saved to: {output_file}")
    print_system_info(info)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            monitor_continuous(interval)
        elif sys.argv[1] == "save":
            output_file = sys.argv[2] if len(sys.argv) > 2 else "vm_snapshot.json"
            save_snapshot(output_file)
        else:
            print("Usage:")
            print("  python vm_monitor.py                 # Single snapshot")
            print("  python vm_monitor.py monitor [30]    # Continuous monitoring")
            print("  python vm_monitor.py save [file.json] # Save snapshot to file")
    else:
        # Single snapshot
        info = get_system_info()
        print_system_info(info)
