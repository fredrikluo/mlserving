import json
from annoy import AnnoyIndex

filename = 'test.ann'
index2idfilename = 'index2id.json'

def main():
     # build the ann index file
    t = AnnoyIndex(2, 'dot')  # Length of item vector that will be indexed
    id2index = {}
    index2id = {}

    embeddings = [[0.4472135954999579, 0.8944271909999159],
                  [0.4472135954999579, 0.8944271909999159],
                  [0.8944271909999159, 0.4472135954999579]]
    for i, v in enumerate(embeddings):
        t.add_item(i, v)
        id2index[str(i)] = i
        index2id[i] = str(i)
    t.build(1)
    t.save(filename)

    with open(index2idfilename, 'w') as mapping:
        json.dump(index2id, mapping, indent=4)


if __name__ == '__main__':
    main()