import time

def prefix(pattern):
    m = len(pattern)
    pi = [0] * m
    k = 0
    for q in range(1, m):
        while k > 0 and pattern[k] != pattern[q]:
            k = pi[k - 1]
        if pattern[k] == pattern[q]:
            k += 1
        pi[q] = k
    return pi


def naive_match(text, pattern):
    n = len(text)
    m = len(pattern)
    comparisons = 0
    for i in range(n - m + 1):
        is_match = True
        for j in range(m):
            comparisons += 1
            if text[i + j] != pattern[j]:
                is_match = False
                break
        if is_match:
            return i, comparisons
    return -1, comparisons


def boyer_moore_horspool_match(text, pattern):
    n = len(text)
    m = len(pattern)
    d = {}
    unique = set()
    for i in range(m - 2, -1, -1):
        if pattern[i] not in unique:
            d[pattern[i]] = m - i - 1
            unique.add(pattern[i])
    if pattern[m - 1] not in unique:
        d[pattern[m - 1]] = m
    d['*'] = m
    comparisons = 0
    if n >= m:
        i = m - 1
        while i < n:
            k = 0
            j = 0
            for j in range(m - 1, -1, -1):
                comparisons += 1
                if text[i - k] != pattern[j]:
                    if j == m - 1:
                        dis = d[text[i]] if d.get(text[i], False) else d['*']
                    else:
                        dis = d[pattern[j]]
                    i += dis
                    break
                k += 1
            if j == 0:
                return i - k + 1, comparisons
    return -1, comparisons


def rabin_karp_match(text, pattern):
    n = len(text)
    m = len(pattern)
    p = 31
    comparisons = 0
    hash_pattern = 0
    for i in range(m):
        hash_pattern += ord(pattern[i]) * (p ** (m - 1 - i))
    hash_text = 0
    for i in range(m):
        hash_text += ord(text[i]) * (p ** (m - 1 - i))
    for i in range(n - m + 1):
        if hash_text == hash_pattern:
            for j in range(m):
                comparisons += 1
                if text[i + j] != pattern[j]:
                    break
            if j == m - 1:
                return i, comparisons
        if i < n - m:
            hash_text = (
                (hash_text - ord(text[i]) * (p ** (m - 1))) * p + ord(text[i + m])
            )
    return -1, comparisons


def kmp_match(text, pattern):
    n = len(text)
    m = len(pattern)
    pi = prefix(pattern)
    q = 0
    comparisons = 0
    for i in range(n):
        while q > 0 and pattern[q] != text[i]:
            q = pi[q - 1]
            comparisons += 1
        if pattern[q] == text[i]:
            q += 1
            comparisons += 1
        if q == m:
            return i - m + 1, comparisons
    return -1, comparisons


def naive_search(text_file, pattern_file):
    with open(text_file, 'r', encoding="utf8") as text_f:
        text = text_f.read()
    with open(pattern_file, 'r', encoding="utf8") as pattern_f:
        pattern = pattern_f.read()

    start_time = time.perf_counter()

    index, comparisons = naive_match(text, pattern)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return index, comparisons, execution_time

def boyer_moore_horspool_search(text_file, pattern_file):
    with open(text_file, 'r', encoding="utf8") as text_f:
        text = text_f.read()
    with open(pattern_file, 'r', encoding="utf8") as pattern_f:
        pattern = pattern_f.read()

        start_time = time.perf_counter()

        index, comparisons = boyer_moore_horspool_match(text, pattern)

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return index, comparisons, execution_time


def rabin_karp_search(text_file, pattern_file):
    with open(text_file, 'r', encoding="utf8") as text_f:
        text = text_f.read()
    with open(pattern_file, 'r', encoding="utf8") as pattern_f:
        pattern = pattern_f.read()

    start_time = time.perf_counter()

    index, comparisons = rabin_karp_match(text, pattern)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return index, comparisons, execution_time


def kmp_search(text_file, pattern_file):
    with open(text_file, 'r', encoding="utf8") as text_f:
        text = text_f.read()
    with open(pattern_file, 'r', encoding="utf8") as pattern_f:
        pattern = pattern_f.read()

    start_time = time.perf_counter()

    index, comparisons = kmp_match(text, pattern)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return index, comparisons, execution_time


def result(index, comparisons, execution_time):
    if index != -1:
        print(f"Шаблон найден в позиции {index} в тексте.")
    else:
        print("Шаблон не найден в тексте.")
    print(f"Количество операций сравнения строк: {comparisons}")
    print(f"Время выполнения алгоритма: {round(execution_time, 5)} секунд\n")


def main():
    for i in range(1, 5):
        text_file = f"tests/good_t_{i}.txt"
        pattern_file = f"tests/good_w_{i}.txt"
        print(f"good test {i}")

        execution_time_mean = 0
        print("Наивный алгоритм")
        for j in range(1, 15):
            index, comparisons, execution_time = naive_search(text_file, pattern_file)
            execution_time_mean += execution_time
        result(index, comparisons, execution_time_mean / 15)
        execution_time_mean = 0
        print("Алгоритм Бойера-Мура-Хорспула")
        for j in range(1, 15):
            index, comparisons, execution_time = boyer_moore_horspool_search(text_file, pattern_file)
            execution_time_mean += execution_time
        result(index, comparisons, execution_time_mean / 15)
        execution_time_mean = 0
        print("Алгоритм Рабина-Карпа")
        for j in range(1, 15):
            index, comparisons, execution_time = rabin_karp_search(text_file, pattern_file)
            execution_time_mean += execution_time
        result(index, comparisons, execution_time_mean / 15)
        execution_time_mean = 0
        print("Алгоритм Кнута-Морриса-Пратта")
        for j in range(1, 15):
            index, comparisons, execution_time = kmp_search(text_file, pattern_file)
            execution_time_mean += execution_time
        result(index, comparisons, execution_time_mean / 15)

    for i in range(1, 5):
        text_file = f"tests/bad_t_{i}.txt"
        pattern_file = f"tests/bad_w_{i}.txt"
        print(f"bad test {i}")

        execution_time_mean = 0
        print("Наивный алгоритм")
        for j in range(1, 15):
            index, comparisons, execution_time = naive_search(text_file, pattern_file)
            execution_time_mean += execution_time
        result(index, comparisons, execution_time_mean / 15)
        execution_time_mean = 0
        print("Алгоритм Бойера-Мура-Хорспула")
        for j in range(1, 15):
            index, comparisons, execution_time = boyer_moore_horspool_search(text_file, pattern_file)
            execution_time_mean += execution_time
        result(index, comparisons, execution_time_mean / 15)
        execution_time_mean = 0
        print("Алгоритм Рабина-Карпа")
        for j in range(1, 15):
            index, comparisons, execution_time = rabin_karp_search(text_file, pattern_file)
            execution_time_mean += execution_time
        result(index, comparisons, execution_time_mean / 15)
        execution_time_mean = 0
        print("Алгоритм Кнута-Морриса-Пратта")
        for j in range(1, 15):
            index, comparisons, execution_time = kmp_search(text_file, pattern_file)
            execution_time_mean += execution_time
        result(index, comparisons, execution_time_mean / 15)


if __name__ == "__main__":
    main()