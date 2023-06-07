import pandas as pd

# Read the data from a .csv file:
results = pd.read_csv("results/complete_results.csv", sep=";", chunksize=1000)

for chunk in results:
    print(chunk.shape)
    for index, row in chunk.iterrows():
        controversiality = row[2]
        text = row[1]
        prediction = str(row[5]).replace("\n      ", "").replace("[array([", "").replace(".])]", "").split("., ")
        prediction = [float(value) for value in prediction]

        print(f"{controversiality} {prediction}: {text}")