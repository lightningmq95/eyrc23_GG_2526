def calcu(n):
    for i in range(n):
        while i!=0:
            if i%2==0:
                print(f"{2*i}")
            else:
                print(f"{i**2}")
            break
        else:
            print(f"{i+3}")
T=int(input())
for item in range(T):
    n=int(input())
    calcu(n)
    
