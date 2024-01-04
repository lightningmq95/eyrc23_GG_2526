def patterns(N):
    for i in reversed(range(N,0,-1)):
        stars=N-i+1
        hashes= (stars-1)//5
        for j in range(stars):
            if (j + 1) % 5 == 0:
                print("#", end="")
            else:
                print("*", end="")
        print()
T=int(input())
for items in range(T):
    N=int(input())
    patterns(N)