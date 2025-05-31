R, C = map(int, input().split())
field = [list(input()) for _ in range(R)]

dx = [-1, 1, 0, 0]  # Смещения по строкам: вверх, вниз
dy = [0, 0, -1, 1]  # Смещения по столбцам: влево, вправо

ok = True

for i in range(R):
    for j in range(C):
        if field[i][j] == 'S':
            for d in range(4):
                ni = i + dx[d]
                nj = j + dy[d]
                if 0 <= ni < R and 0 <= nj < C:
                    if field[ni][nj] == 'W':
                        ok = False  # Волк рядом — спасти нельзя
                    elif field[ni][nj] == '.':
                        field[ni][nj] = 'D'  # Ставим собаку для защиты

if not ok:
    print("No")
else:
    print("Yes")
    for row in field:
        print("".join(row))
