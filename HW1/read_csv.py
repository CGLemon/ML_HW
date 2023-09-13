import pandas as pd
import matplotlib.pyplot as plt

TAGS = ["Sex", "Age", "HR", "Height", "Weight", "BP"]
NUMBER_TAGS = ["Age", "HR", "Height", "Weight", "BP"]

def check_int(v):
    return len(v) != 0 and v.isnumeric()

def transform(df):
    result = dict()
    for i in range(len(df)):
        line = list(df.loc[i])
        name = line[0]
        tag = line[1]
        val = line[2]
        if result.get(name) is None:
            result[name] = dict()
        result[name][tag] = val

    csv_table = {"Name" : list()}
    for t in TAGS:
        csv_table[t] = list()

    for name, v in result.items():
        csv_table["Name"].append(name)
        for t in TAGS:
            csv_table[t].append(v.get(t, str()))
    return pd.DataFrame(csv_table)

def average(df):
    # 計算出 Age, HR, Height, Weight 及 BP 的平均值加於 dataframe 最下
    # 列。（請注意避免累加過程中出現 overflow 現象）

    accm_row = {"Name" : "Avg"}
    for t in TAGS:
        accm_row[t] = str() 

    for t in NUMBER_TAGS:
        cnt = 0
        accm = 0
        for v in df[t]:
            if len(v) != 0:
                accm += int(v)
                cnt += 1
        accm_row[t] = "{:.2f}".format(accm/cnt)
    df = df._append(accm_row, ignore_index=True)
    return df

def find_max(df):
    # 利用程式找出並於螢幕列出各分項指標 feature（Age, HR, Height,
    # Weight 及 BP）中之最大者的姓名（Name）。

    for t in NUMBER_TAGS:
        max_name = list()
        max_val = None
        for idx, v in enumerate(df[t]):
            if check_int(v):
                if max_val is None or int(v) > max_val:
                    max_val = int(v)
                    max_name.append(df["Name"][idx])
                elif int(v) == max_val:
                    max_name.clear()
                    max_name.append(df["Name"][idx])
        print("{} {} {}".format(t, max_name, max_val))

def scatter(df):      
    # 請繪製身高體重（Height, Weigth）之散佈圖（Scatter Plot），女生請以
    # 紅點標示，男生請以藍點標示。      

    def separate(data):
        data_x = list()
        data_y = list()
        for x, y in data:
            data_x.append(x)
            data_y.append(y)
        return data_x, data_y

    data_m = list()
    data_f = list()

    for idx, v in enumerate(df["Sex"]):
        w = df["Weight"][idx]
        h = df["Height"][idx]
        if check_int(w) and check_int(h):
            w = int(w)
            h = int(h)
            if v == "M":
                data_m.append((w, h))
            elif v == "F":
                data_f.append((w, h))

    data_m_x, data_m_y = separate(data_m)
    data_f_x, data_f_y = separate(data_f)
    plt.scatter(data_m_x, data_m_y, c='b')
    plt.scatter(data_f_x, data_f_y, c='r')
    plt.show()

def pie_and_bar(df): 
    # 繪製不同年齡（Age）區間的人數直方圖及圓餅圖（Pie Chart）。請依
    # 1-10, 11-20, 21-30…每 10 歲為一個區間統計。

    def as_data(ages):
        x = list()
        y = list()
        labels = list()
        keys = sorted(list(ages.keys()))
        for k in keys:
            x.append(k+1)
            y.append(ages[k])
            labels.append("{}-{}".format(10*k+1, 10*(k+1)))
        return x, y, labels

    ages = dict()
    for idx, v in enumerate(df["Age"]):
        if check_int(v):
            level = (int(v)-1) // 10
            if not level in ages:
                ages[level] = 0
            ages[level] += 1

    x, y, labels = as_data(ages)
    plt.pie(y, labels=labels, radius=1.5)
    plt.show()
    plt.bar(x, y, tick_label=labels)
    plt.show()

def sex_pie(df):
    # 繪製男女性別分佈比例之圓餅圖（Pie Chart）。

    def as_data(sex):
        x = [sex["M"], sex["F"]]
        labels = ["Male", "Female"]
        return x, labels

    sex = { "M" : 0 , "F" : 0 }
    for v in df["Sex"]:
        if len(v) != 0:
            sex[v] += 1

    x, labels = as_data(sex)
    plt.pie(x, labels=labels, radius=1.5)
    plt.show()

def main():
    df = transform(pd.read_csv("HW1_data.csv"))
    df = average(df)

    find_max(df)
    scatter(df)
    pie_and_bar(df)
    sex_pie(df)

if __name__ == '__main__':
    main()
