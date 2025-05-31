n = int(input())
f = list(map(int, input().split()))

# Приведем к индексам с нуля
f = [x - 1 for x in f]

found = False

for i in range(n):
    j = f[i]
    k = f[j]
    if f[k] == i:
        found = True
        break

print("YES" if found else "NO")
