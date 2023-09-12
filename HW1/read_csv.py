import pandas as pd

# Name, Sex, Age, HR, Height, Weight, BP

def main():
    df = pd.read_csv("HW1_data.csv")

    result = dict()

    for i in range(len(df)):
        line = list(df.loc[i])
        name = line[0]
        tag = line[1]
        val = line[2]
        if result.get(name) is None:
            result[name] = dict()
        result[name][tag] = val

    tags = ["Sex", "Age", "HR", "Height", "Weight", "BP"]

    csv_table = {"Name" : list()}
    for t in tags:
        csv_table[t] = list()

    for name, v in result.items():
        csv_table["Name"].append(name)
        for t in tags:
            csv_table[t].append(v.get(t, str()))


    df = pd.DataFrame(csv_table)
    accm_row = {"Name" : str()}
    for t in tags:
        accm_row[t] = str() 

    for t in ["Age", "HR", "Height", "Weight", "BP"]:
        cnt = 0
        accm = 0
        for i in df[t]:
            if len(i) != 0:
                accm += int(i)
                cnt += 1
        accm_row[t] = "{:.2f}".format(accm/cnt)

    # 計算出 Age, HR, Height, Weight 及 BP 的平均值加於 dataframe 最下
列。（請注意避免累加過程中出現 overflow 現象）
    df = df._append(accm_row, ignore_index=True)

if __name__ == '__main__':
    main()
