n, t = map(int, input().split())
a = list(map(int, input().split()))

position = 1  # начинаем с 1-й ячейки

while position < t:
    position += a[position - 1]  # прыгаем по порталу

print("YES" if position == t else "NO")
