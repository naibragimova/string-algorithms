import time
import random
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(100000)
unicode_chars = ''.join(chr(i) for i in range(0x10FFFF + 1) if chr(i).isprintable() and not chr(i) == '#')


def z_function(string):
    n = len(string)
    z = [0] * n
    z[0] = n
    l = 0
    r = 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and string[z[i]] == string[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1
    return z


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
    results = []
    for i in range(n - m + 1):
        is_match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                is_match = False
                break
        if is_match:
            results.append(i)
    return results


def z_function_match(text, pattern):
    zf = z_function(pattern + '#' + text)
    n = len(text)
    m = len(pattern)
    results = []
    for i in range(m + 1, n + 2):
        if zf[i] == m:
            results.append(i - m - 1)
    return results


def boyer_moore_horspool_match(text, pattern):
    n = len(text)
    m = len(pattern)
    results = []
    d = {}
    unique = set()
    for i in range(m - 2, -1, -1):
        if pattern[i] not in unique:
            d[pattern[i]] = m - i - 1
            unique.add(pattern[i])
    if pattern[m - 1] not in unique:
        d[pattern[m - 1]] = m
    d['*'] = m
    if n >= m:
        i = m - 1
        while i < n:
            k = 0
            j = 0
            for j in range(m - 1, -1, -1):
                if text[i - k] != pattern[j]:
                    if j == m - 1:
                        dis = d[text[i]] if d.get(text[i], False) else d['*']
                    else:
                        dis = d[pattern[j]]
                    i += dis
                    break
                k += 1
            if j == 0:
                results.append(i - m + 1)
                i += d.get(pattern, m)
    return results


def rabin_karp_match(text, patterns):
    n = len(text)
    m = len(patterns[0])
    p = 4999
    q = 10007
    results = {pattern: [] for pattern in patterns}

    pattern_hashes = {pattern: 0 for pattern in patterns}
    for pattern in patterns:
        hash_pattern = 0
        for i in range(m):
            hash_pattern += ord(pattern[i]) * (p ** (m - 1 - i))
        pattern_hashes[pattern] = hash_pattern % q
    hash_text = 0
    for i in range(m):
        hash_text += ord(text[i]) * (p ** (m - 1 - i))
    hash_text %= q
    for i in range(n - m + 1):
        if hash_text in pattern_hashes.values():
            pattern = list(pattern_hashes.keys())[list(pattern_hashes.values()).index(hash_text)]
            for j in range(m):
                if text[i + j] != pattern[j]:
                    break
            if j == m - 1:
                results[pattern].append(i)
        if i < n - m:
            hash_text = ((hash_text - ord(text[i]) * (p ** (m - 1))) * p + ord(text[i + m])) % q
    return results


def kmp_match(text, pattern):
    n = len(text)
    m = len(pattern)
    results = []
    pi = prefix(pattern)
    q = 0
    for i in range(n):
        while q > 0 and pattern[q] != text[i]:
            q = pi[q - 1]
        if pattern[q] == text[i]:
            q += 1
        if q == m:
            results.append(i - m + 1)
            q = pi[q - 1]
    return results


def aho_corasick_match(text, patterns):
    class Node:
        def __init__(self):
            self.son = {}
            self.go = {}
            self.parent = None
            self.suffLink = None
            self.up = None
            self.charToParent = None
            self.isLeaf = False
            self.leafPatternNumber = []

    class AhoCorasick:
        def __init__(self):
            self.root = Node()
            self.root.suffLink = self.root
            self.root.up = self.root

        def getSuffLink(self, node):
            if node.suffLink is None:
                if node == self.root or node.parent == self.root:
                    node.suffLink = self.root
                else:
                    node.suffLink = self.getLink(self.getSuffLink(node.parent), node.charToParent)
            return node.suffLink

        def getLink(self, node, c):
            if c not in node.go:
                if c in node.son:
                    node.go[c] = node.son[c]
                else:
                    if node == self.root:
                        node.go[c] = self.root
                    else:
                        node.go[c] = self.getLink(self.getSuffLink(node), c)
            return node.go[c]

        def getUp(self, node):
            if node.up is None:
                suffLink = self.getSuffLink(node)
                if suffLink.isLeaf:
                    node.up = suffLink
                elif suffLink == self.root:
                    node.up = self.root
                else:
                    node.up = self.getUp(suffLink)
            return node.up

        def addString(self, s, patternNumber):
            cur = self.root
            for char in s:
                if char not in cur.son:
                    new_node = Node()
                    new_node.parent = cur
                    new_node.charToParent = char
                    cur.son[char] = new_node
                cur = cur.son[char]
            cur.isLeaf = True
            cur.leafPatternNumber.append(patternNumber)

        def processText(self, text):
            cur = self.root
            results = {i: [] for i in range(len(patterns))}

            for i in range(len(text)):
                cur = self.getLink(cur, text[i])
                temp = cur
                while temp != self.root:
                    if temp.isLeaf:
                        for patternNumber in temp.leafPatternNumber:
                            results[patternNumber].append(i - len(patterns[patternNumber]) + 1)
                    temp = self.getUp(temp)

            return results

    ac = AhoCorasick()
    for i, pattern in enumerate(patterns):
        ac.addString(pattern, i)
    results = ac.processText(text)
    return results


def naive_search(text, patterns):
    start_time = time.perf_counter()
    indexes = []

    for pattern in patterns:
        indexes.append(naive_match(text, pattern))

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return indexes, execution_time


def z_function_search(text, patterns):
    start_time = time.perf_counter()
    indexes = []

    for pattern in patterns:
        indexes.append(z_function_match(text, pattern))

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return indexes, execution_time


def boyer_moore_horspool_search(text, patterns):
    start_time = time.perf_counter()
    indexes = []

    for pattern in patterns:
        indexes.append(boyer_moore_horspool_match(text, pattern))

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return indexes, execution_time


def rabin_karp_search(text, patterns):
    start_time = time.perf_counter()

    indexes = rabin_karp_match(text, patterns)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return indexes, execution_time


def kmp_search(text, patterns):
    start_time = time.perf_counter()
    indexes = []

    for pattern in patterns:
        indexes.append(kmp_match(text, pattern))

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return indexes, execution_time


def aho_corasick_search(text, patterns):
    start_time = time.perf_counter()

    indexes = aho_corasick_match(text, patterns)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return indexes, execution_time


def result(index, execution_time):
    if index != -1:
        print(f"Шаблон найден в позиции {index} в тексте.")
    else:
        print("Шаблон не найден в тексте.")
    print(f"Время выполнения алгоритма: {round(execution_time, 5)} секунд\n")


def generate_random_string(length_text, length_pattern, num_alphabet, count_patterns):
    selected_chars = unicode_chars[:num_alphabet]
    text = ''.join(random.choices(selected_chars, k=length_text))
    patterns = []
    for _ in range(count_patterns):
        ind1 = random.randint(0, length_text - length_pattern)
        ind2 = ind1 + length_pattern
        patterns.append(text[ind1:ind2])
    return text, patterns


def generate_random_string_with_no_patterns(length_text, length_pattern, num_alphabet):
    selected_chars = unicode_chars[:num_alphabet]
    text = ''.join(random.choices(selected_chars, k=length_text))
    patterns = []
    pattern = ''.join(random.choices(selected_chars, k=length_pattern))
    while (pattern in text):
        pattern = ''.join(random.choices(selected_chars, k=length_pattern))
    patterns.append(pattern)
    return text, patterns


def test_with_random_string(params, count_test, count_patterns):
    text_pattern_alphabet_params = params
    results = []
    test_num = 0
    for i in text_pattern_alphabet_params:
        results.append({'Наивный алгоритм': 0,
                        'Поиск с помощью Z-функции': 0,
                        'Алгоритм Бойера-Мура-Хорспула': 0,
                        'Алгоритм Рабина-Карпа': 0,
                        'Алгоритм Кнута-Морриса-Пратта': 0,
                        'Алгоритм Ахо-Корасик': 0})
        for _ in range(count_test):
            if count_patterns == 0:
                text, patterns = generate_random_string_with_no_patterns(i[0], i[1], i[2])
            else:
                text, patterns = generate_random_string(i[0], i[1], i[2], count_patterns)

            indexes, execution_time = naive_search(text, patterns)
            results[test_num]['Наивный алгоритм'] += execution_time

            indexes, execution_time = z_function_search(text, patterns)
            results[test_num]['Поиск с помощью Z-функции'] += execution_time

            indexes, execution_time = boyer_moore_horspool_search(text, patterns)
            results[test_num]['Алгоритм Бойера-Мура-Хорспула'] += execution_time

            indexes, execution_time = rabin_karp_search(text, patterns)
            results[test_num]['Алгоритм Рабина-Карпа'] += execution_time

            indexes, execution_time = kmp_search(text, patterns)
            results[test_num]['Алгоритм Кнута-Морриса-Пратта'] += execution_time

            indexes, execution_time = aho_corasick_search(text, patterns)
            results[test_num]['Алгоритм Ахо-Корасик'] += execution_time

            results[test_num]['Наивный алгоритм'] = results[test_num]['Наивный алгоритм'] / count_test
            results[test_num]['Поиск с помощью Z-функции'] = results[test_num]['Поиск с помощью Z-функции'] / count_test
            results[test_num]['Алгоритм Бойера-Мура-Хорспула'] = results[test_num][
                                                                     'Алгоритм Бойера-Мура-Хорспула'] / count_test
            results[test_num]['Алгоритм Рабина-Карпа'] = results[test_num]['Алгоритм Рабина-Карпа'] / count_test
            results[test_num]['Алгоритм Кнута-Морриса-Пратта'] = results[test_num][
                                                                     'Алгоритм Кнута-Морриса-Пратта'] / count_test
            results[test_num]['Алгоритм Ахо-Корасик'] = results[test_num]['Алгоритм Ахо-Корасик'] / count_test

        test_num += 1

    for i in range(len(results)):
        # print(f"Длина строки - {text_length[i]}, длина паттерна - {pattern_length[i]}")
        print(", ".join(f"{a}: {b:.6f} сек" for a, b in results[i].items()))

    return results


def plot_graphs(results, params):
    text_lengths = []
    pattern_length_ratios = [0.1, 0.5, 0.9]
    alphabet_sizes = []

    for param in params:
        if not param[0] in text_lengths:
            text_lengths.append(param[0])
        if not param[2] in alphabet_sizes:
            alphabet_sizes.append(param[2])

    algorithms = ['Наивный алгоритм', 'Поиск с помощью Z-функции', 'Алгоритм Бойера-Мура-Хорспула',
                  'Алгоритм Рабина-Карпа', 'Алгоритм Кнута-Морриса-Пратта', 'Алгоритм Ахо-Корасик']
    for alphabet_size in alphabet_sizes:
        for pattern_length_ratio in pattern_length_ratios:
            fig, ax = plt.subplots()
            for algorithm in algorithms:
                times = []
                for text_length in text_lengths:
                    if [text_length, int(pattern_length_ratio * text_length), alphabet_size] in params:
                        i = params.index([text_length, int(pattern_length_ratio * text_length), alphabet_size])
                        times.append(results[i][algorithm])
                ax.plot(text_lengths, times, label=algorithm)

            title = f'Анализ времени выполнения, мощность алфавита = {alphabet_size},'
            title += f' длина паттерна = {pattern_length_ratio} от длины строки'
            ax.set_title(title)
            ax.set_xlabel('Длина строки')
            ax.set_ylabel('Время выполнения (сек)')
            ax.legend()
            plt.show()


def main():
    params = [[i, j, k] for i in range(10, 1000, 150) for j in [int(0.1 * i), int(0.5 * i), int(0.9 * i)] for k in [10, 100, 1000]]
    results = test_with_random_string(params, 50, 0)
    plot_graphs(results, params)


if __name__ == "__main__":
    main()