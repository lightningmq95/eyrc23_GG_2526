T = int(input())

for _ in range(T):
    input_str = input()
    input_str = input_str[1:]
    words = input_str.split()
    word_length = [len(word) for word in words]
    result_str = ",".join(map(str, word_length))
    print(result_str)
