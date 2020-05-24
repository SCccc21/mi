from work import load_iwpc, extract_target, inver, engine

if __name__ == "__main__":
    data_folder = 'data'
    x, y, featnames = load_iwpc(data_folder)
    col = range(2, 8)
    for i in col:
    	print(x[:, i].sum())
	