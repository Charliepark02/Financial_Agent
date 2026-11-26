from concurrent.futures import ThreadPoolExecutor
from time import perf_counter


def execute_batch(registry, calls, max_workers=4):
    """Execute independent (name, arguments) pairs in submission order."""
    def run(call):
        name, arguments = call
        start = perf_counter()
        result = registry.execute(name=name, arguments=arguments)
        return result, perf_counter() - start

    if len(calls) < 2:
        return list(map(run, calls))
    with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as pool:
        return list(pool.map(run, calls))
