n = int(input())
p = [int(input()) for _ in range(n)]

depth = [0] * n
max_depth = 0

for i in range(n):
    j = i
    current_depth = 0

    # Поднимаемся по цепочке начальников
    while j != -1:
        if depth[j] > 0:
            current_depth += depth[j]
            break
        current_depth += 1
        j = p[j] - 1 if p[j] != -1 else -1

    # Второй проход — сохраняем глубины для всех в цепочке
    j = i
    while j != -1 and depth[j] == 0:
        depth[j] = current_depth
        current_depth -= 1
        j = p[j] - 1 if p[j] != -1 else -1

    max_depth = max(max_depth, depth[i])

print(max_depth)
