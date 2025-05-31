n = int(input())
p = list(map(int, input().split()))

res = []

for start in range(n):
    visited = set()
    curr = start
    while curr not in visited:
        visited.add(curr)
        curr = p[curr] - 1  # -1 для приведения к индексации с 0
    res.append(curr + 1)  # +1 чтобы вернуть к номерам школьников от 1

print(*res)
