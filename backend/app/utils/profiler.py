from line_profiler import LineProfiler
import functools
import sys
import asyncio

def profile_sync(func):
    """Use this on synchronous methods (e.g., inside DataManager or _assemble_and_process_zip)"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        # Add the target function to the profiler
        lp_wrapper = lp(func)
        
        # Execute the function
        result = lp_wrapper(*args, **kwargs)
        
        # Print the results to the Uvicorn console
        print(f"\n\n{'='*60}\nLINE PROFILER REPORT FOR: {func.__name__}\n{'='*60}")
        lp.print_stats(stream=sys.stdout)
        return result
    return wrapper