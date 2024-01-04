def dec_to_binary(n, binary_result=""):
    if n == 0:
        binary_result = binary_result.rjust(8, '0')
        print(binary_result)
        return
    remainder = n % 2
    binary_result = str(remainder) + binary_result
    dec_to_binary(n // 2, binary_result)
T = int(input())
for _ in range(T):
    n = int(input())
    dec_to_binary(n)