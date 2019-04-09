qq = [77,73,83,81,83,81,83,87]
result = [
    chr(value) for key,value in enumerate(qq)
    if ((lambda key: True if (key & (key-1)) == 0 else False)(key))]
print(result)
