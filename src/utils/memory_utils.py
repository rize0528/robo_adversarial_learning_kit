"""Memory management utilities for VLM model loading."""

import psutil
import torch
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def get_memory_info() -> Dict[str, float]:
    """Get current system memory information.
    
    Returns:
        Dict containing memory information in GB
    """
    memory = psutil.virtual_memory()
    
    memory_info = {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3), 
        "used_gb": memory.used / (1024**3),
        "percent_used": memory.percent
    }
    
    # Add GPU memory if CUDA is available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0)
        memory_info.update({
            "gpu_total_gb": gpu_memory.total_memory / (1024**3),
            "gpu_reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
            "gpu_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3)
        })
    
    return memory_info


def check_memory_requirements(required_memory_gb: float, buffer_gb: float = 2.0) -> Tuple[bool, str]:
    """Check if system has enough memory for model loading.
    
    Args:
        required_memory_gb: Memory required for model in GB
        buffer_gb: Additional buffer memory to keep free
        
    Returns:
        Tuple of (can_load, reason_message)
    """
    memory_info = get_memory_info()
    available_memory = memory_info["available_gb"]
    total_needed = required_memory_gb + buffer_gb
    
    if available_memory >= total_needed:
        return True, f"Sufficient memory: {available_memory:.1f}GB available, {total_needed:.1f}GB needed"
    else:
        return False, f"Insufficient memory: {available_memory:.1f}GB available, {total_needed:.1f}GB needed"


def log_memory_usage(stage: str = "current"):
    """Log current memory usage for debugging."""
    memory_info = get_memory_info()
    
    logger.info(f"Memory usage at {stage}:")
    logger.info(f"  System: {memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB ({memory_info['percent_used']:.1f}%)")
    
    if "gpu_total_gb" in memory_info:
        gpu_used = memory_info.get("gpu_allocated_gb", 0)
        gpu_total = memory_info["gpu_total_gb"]
        logger.info(f"  GPU: {gpu_used:.1f}GB / {gpu_total:.1f}GB")


def clear_gpu_memory():
    """Clear GPU memory if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU memory cache cleared")