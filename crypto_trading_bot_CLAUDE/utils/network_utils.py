"""
Utilities for robust network communication with API endpoints
"""
import time
import random
import functools
import requests
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from utils.logger import setup_logger

logger = setup_logger("network_utils")

# Type variable for decorators
T = TypeVar('T')

def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    errors=(requests.exceptions.RequestException, ConnectionError, TimeoutError)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff for network operations
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplicative factor for delay after each retry
        errors: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            last_exception = None
            
            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    last_exception = e
                    
                    if retry >= max_retries:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        break
                    
                    # Add jitter to avoid thundering herd problem
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = min(delay + jitter, max_delay)
                    
                    logger.warning(
                        f"Retry {retry+1}/{max_retries} due to {e.__class__.__name__}: {str(e)}. "
                        f"Waiting {sleep_time:.2f}s before next attempt"
                    )
                    
                    time.sleep(sleep_time)
                    delay = min(delay * backoff_factor, max_delay)
            
            # If all retries failed, raise the last exception
            if last_exception:
                raise last_exception
                
            return cast(T, None)  # This should never be reached
            
        return wrapper
    return decorator

class APIRateLimiter:
    """
    Rate limiter to prevent API rate limit errors by tracking request frequency
    """
    def __init__(self, requests_per_min: int = 60, request_window: float = 60.0):
        self.requests_per_min = requests_per_min
        self.request_window = request_window
        self.request_timestamps = []
    
    def wait_if_needed(self):
        """
        Waits if necessary to respect rate limits
        """
        current_time = time.time()
        
        # Remove timestamps outside the current window
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts <= self.request_window]
        
        # If we've hit the rate limit, wait until we can make another request
        if len(self.request_timestamps) >= self.requests_per_min:
            oldest_timestamp = self.request_timestamps[0]
            sleep_time = oldest_timestamp + self.request_window - current_time + 0.1
            
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.2f}s before next request")
                time.sleep(sleep_time)
        
        # Record this request
        self.request_timestamps.append(time.time())

# Initialize a global rate limiter for Binance API
binance_rate_limiter = APIRateLimiter(requests_per_min=1000, request_window=60.0)
