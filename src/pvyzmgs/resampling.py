import numpy as np


def bootstrap(f, values, samples=1000):
    n = len(values)
    resampled = [f(np.random.choice(values, n, replace=True)) for _ in range(samples)]
    return np.std(resampled)


def jackknife(f, values):
    n = len(values)
    resampled = []
    for i in range(n):
        temp = list(values)
        del temp[i]
        resampled.append(f(temp))

    val = f(values)
    return np.sqrt(np.sum([(rs - val) ** 2 for rs in resampled]))


def main():
    values = np.random.randint(0, 100, 100)
    f = lambda vs: np.var(vs)

    print(f"value: {f(values)}")
    for samples in [5, 10, 50, 100, 500, 1000, 5000]:
        results = ", ".join(
            str(bootstrap(f, values, samples=samples)) for _ in range(5)
        )
        print(f"bootstrap (samples={samples}): {results}")
    print(f"jackknife: {jackknife(f, values)}")


if __name__ == "__main__":
    main()
