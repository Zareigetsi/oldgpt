import urllib.request, numpy as np, pickle

def fetch(book_id):
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    t = urllib.request.urlopen(url).read().decode("utf-8")
    s = t.find("\n", t.find("*** START OF")) + 1   # skip header
    e = t.find("*** END OF")                       # drop footer
    return t[s:e].strip()

ids = 	[2701]  # add more IDs here to combine books
text = "\n\n\n".join(fetch(i) for i in ids)       # separator between books
text = text.replace("\r\n", "\n")

with open("input.txt", "w", encoding="utf-8") as f:
    f.write(text)

chars = sorted(list((set(text))))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.uint16)


n = int(0.9 * len(data))
data[:n].tofile("train.bin"); data[n:].tofile("val.bin")
pickle.dump({"stoi": stoi, "itos": {i: c for c, i in stoi.items()}}, open("meta.pkl", "wb"))
print(len(text), "chars, vocab", len(chars))
print(''.join(chars))
print(len(chars))