import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
if __name__ == "__main__":
    df1 = pd.read_csv("o100.csv", sep = ',', header = None)
    df1 = df1.drop(columns = [100])
    df1 = np.log(df1)
    df1 = np.array(df1)

    df2 = pd.read_csv("o200.csv", sep = ',', header = None)
    df2 = df2.drop(columns = [200])
    df2 = np.log(df2)
    df2 = np.array(df2)

    df5 = pd.read_csv("o500.csv", sep = ',', header = None)
    df5 = df5.drop(columns = [500])
    df5 = np.log(df5)
    df5 = np.array(df5)

    fig, ax = plt.subplots(1, 3, figsize = (16, 5))
    ax[0].imshow(df1)
    ax[1].imshow(df2)
    ax[2].imshow(df5)
    ax[0].set_title("100$^{3}$, PCS")
    ax[1].set_title("200$^{3}$, PCS")
    ax[2].set_title("500$^{3}$, PCS")
    plt.tight_layout()
    plt.savefig("visual.jpg", dpi = 300)
