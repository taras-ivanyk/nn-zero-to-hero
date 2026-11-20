import tiktoken

for i in (tiktoken.list_encoding_names()): 
    enc = tiktoken.get_encoding(i)
    print(f"Vocab length = {enc.n_vocab}")
    print(f"Encoded \"Hi, my name is Taras\" = {enc.encode("Hi, my name is Taras")}")
    print(f"Decoded \"Hi, my name is Taras\" back = {enc.decode(enc.encode("Hi, my name is Taras"))}")
    print("\n")


