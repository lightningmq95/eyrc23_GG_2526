def palindrome(s):
    s = s.lower()
    if s[:]==s[::-1]:
        print(f"It is a palindrome")
    else:
        print(f"It is not a palindrome")

T = int(input())

for item in range(T):
    str=input()
    palindrome(str)