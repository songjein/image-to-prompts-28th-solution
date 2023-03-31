from preproc_split_dataset import preprocess

if __name__ == "__main__":
    make_meta = False

    cnt = 0
    captions = []
    with open("./resources/openprompts_dedup_075.txt") as f:
        for line in f:
            text = line.strip()
            text = preprocess(text)
            if text is None:
                cnt += 1
                continue
            captions.append(line.strip())

    print("removed", cnt)

    with open("./resources/openprompts_dedup_075_filtered.txt", "w") as f:
        for caption in captions:
            f.write(caption + "\n")
