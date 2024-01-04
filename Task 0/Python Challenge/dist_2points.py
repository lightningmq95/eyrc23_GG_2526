import math

def compute_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

T = int(input())
for _ in range(T):
    x1, y1, x2, y2 = map(int, input().split())
    distance = compute_distance(x1, y1, x2, y2)
    print(f"Distance: {distance:.2f}")