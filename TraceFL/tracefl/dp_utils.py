"""Differential Privacy Utilities for TraceFL."""

import numpy as np
import logging
from flwr.common import NDArrays

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_clip_inputs_inplace(model_update, clipping_norm):
    """Safely clip model updates in-place with handling for zero norm cases.
    
    Args:
        model_update: List of model parameter updates
        clipping_norm: Maximum L2 norm for clipping
    """
    try:
        # Validate inputs
        if not model_update or not isinstance(model_update, list):
            logger.error("Invalid model update format")
            return
            
        if clipping_norm <= 0:
            logger.error(f"Invalid clipping norm: {clipping_norm}")
            return

        # Compute L2 norm of the update
        input_norm = compute_l2_norm(model_update)
        logger.info(f"Input norm before clipping: {input_norm:.6f}")
        
        # Handle zero norm case
        if input_norm == 0:
            logger.warning("Model update has zero norm, skipping clipping")
            return
            
        # Clip if norm exceeds threshold
        if input_norm > clipping_norm:
            scaling_factor = clipping_norm / input_norm
            logger.info(f"Applying clipping with scaling factor: {scaling_factor:.6f}")
            for i in range(len(model_update)):
                model_update[i] = model_update[i] * scaling_factor
            
            # Verify clipping
            final_norm = compute_l2_norm(model_update)
            logger.info(f"Final norm after clipping: {final_norm:.6f}")
                
    except Exception as e:
        logger.error(f"Error in safe_clip_inputs_inplace: {str(e)}")
        raise

def compute_l2_norm(inputs):
    """Compute L2 norm of input arrays.
    
    Args:
        inputs: List of numpy arrays
        
    Returns:
        float: L2 norm of the concatenated arrays
    """
    try:
        if not inputs:
            logger.warning("Empty input list provided to compute_l2_norm")
            return 0.0
            
        # Validate input types
        if not all(isinstance(x, np.ndarray) for x in inputs):
            logger.error("Invalid input type - all inputs must be numpy arrays")
            return 0.0
            
        # Compute norm using the same approach as Flower's get_norm
        array_norms = [np.linalg.norm(array.flat) for array in inputs]
        total_norm = float(np.sqrt(sum([norm**2 for norm in array_norms])))
        return total_norm
        
    except Exception as e:
        logger.error(f"Error computing L2 norm: {str(e)}")
        return 0.0

def get_norm(input_arrays: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    try:
        if not input_arrays:
            logger.warning("Empty input arrays provided to get_norm")
            return 0.0
            
        array_norms = [np.linalg.norm(array.flat) for array in input_arrays]
        total_norm = float(np.sqrt(sum([norm**2 for norm in array_norms])))
        return total_norm
        
    except Exception as e:
        logger.error(f"Error in get_norm: {str(e)}")
        return 0.0 
