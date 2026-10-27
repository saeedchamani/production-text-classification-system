import time

from src.predict import predict_text


TEXT = "I forgot my password"
RUNS = 1000


# Warm-up
for _ in range(20):
    predict_text(TEXT)


times = []

for _ in range(RUNS):
    start = time.perf_counter()

    predict_text(TEXT)

    end = time.perf_counter()

    times.append((end - start) * 1000)


average_latency = sum(times) / len(times)

print(f"Runs: {RUNS}")
print(f"Average latency: {average_latency:.3f} ms")
print(f"Min latency: {min(times):.3f} ms")
print(f"Max latency: {max(times):.3f} ms")