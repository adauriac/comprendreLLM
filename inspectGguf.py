#!/usr/bin/env python
import sys
from gguf import GGUFReader
try:
    from llama_cpp import Llama
    llamaImported = True
except:
    llamaImported = False

filename = sys.argv[1]
reader = GGUFReader(filename)

name = reader.fields["general.name"].contents()
arch = reader.fields["general.architecture"].contents()
print("Nom :", name)
d = reader.fields[f"{arch}.embedding_length"].contents()
print("Dimension embeddings :", d)

tokens = reader.fields["tokenizer.ggml.tokens"].contents()
V = len(tokens)
print("Taille vocabulaire :", V)
while True:
    res=input(f"To see tokens between begin end (<{V}). Enter begin et end (if error stop showing tokens)")
    try:
        b,e=map(int,res.split())
        for i, token in enumerate(tokens[b:e+1]):
            print(i, token)
    except:
        break
q = input("Type a question to see its decomposition in tokens (empty->no) ")
llm = Llama(model_path=filename, vocab_only=True, verbose=False)
ids = llm.tokenize(question.encode("utf-8"), add_bos=False, special=False)
sys.exit(123)

print("Nombre de tenseurs :", len(reader.tensors))
for t in reader.tensors:
    shape = tuple(int(x) for x in t.shape)
    print(f"{t.name:35s} "
          f"{str(shape):20s} "
          f"{t.n_elements:12,d} "
          f"{t.tensor_type.name:10s} "
          f"{t.n_bytes:12,d} octets")
    input("hit return")
