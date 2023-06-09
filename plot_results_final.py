import pandas as pd
import re
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
def contains_removed(x):
    return '[removed]' in str(x)


def contains_deleted(x):
    return '[deleted]' in str(x)


# Get only the numbers occurring in a string, as a list:
def str_to_list(x):
    return [int(s) for s in re.findall(r'\d+', x)]


def read_data():
    # Read the data from a .csv file:
    results = pd.read_csv("results/complete_results.csv", sep=";")

    # Get only the 3 columns of interest:
    cols_of_interest = pd.DataFrame({'cont': results.values[:, 2],
                                     'comment': results.values[:, 1],
                                     'values': results.values[:, 5]})  # Get only columns of interest

    # Remove any rows with NaNs:
    data_without_nans = cols_of_interest.dropna()

    # Remove [deleted], [removed] comments:
    data_without_deleted = data_without_nans[data_without_nans['comment'].map(contains_deleted) != True]  # Remove columns with deleted comments
    clean_data = data_without_deleted[data_without_deleted['comment'].map(contains_removed) != True]  # Remove columns with removed comments

    # Reset indexes after deleting rows:
    clean_data.reset_index(inplace=True)

    cont_scores = [int(score) for score in clean_data['cont']]
    preds = list(map(str_to_list, clean_data['values']))

    return cont_scores, preds

def plot_1(values_count, controversial_count):
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 10))
    values = ["Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
              "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition",
              "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring",
              "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance",
              "Universalism: objectivity"]
    sns.set_color_codes("pastel")
    sns.barplot(x=values_count, y=values,
                label="Total", color="b")

    sns.set_color_codes("muted")
    sns.barplot(x=controversial_count, y=values,
                label="Controversial", color="b")

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="Values",
           xlabel="Instances in dataset")
    sns.despine(left=True, bottom=True)

    plt.title("Comparison of controversial and total instances of each value found in CMV dataset")
    plt.tight_layout()
    plt.savefig('plots/plot1_400_filtered_final.png')
    plt.show()

def plot_4(values_count, controversial_count):
    percentage_cont = []
    for i in range(0, len(values_count)):
        if values_count[i] == 0:
            percentage_cont.append(0)
        else:
            percentage_cont.append((controversial_count[i] / values_count[i]) * 100)
    print(percentage_cont)

    average_cont = numpy.sum(controversial_count) / numpy.sum(values_count) * 100

    f, ax = plt.subplots(figsize=(10, 10))
    values = ["Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
              "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition",
              "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring",
              "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance",
              "Universalism: objectivity"]

    plt.axvline(x=average_cont, color='r', linestyle='dashed', label='Average percentage controversial')
    plt.legend(loc="upper right")
    ax.barh(values, percentage_cont)

    ax.set(ylabel="Values",
           xlabel="Percentage Controversial")
    plt.title("Percentage of instances of values which are controversial")
    plt.tight_layout()
    plt.savefig('plots/plot_2_400_filtered_final.png')
    plt.show()


cont_scores, preds = read_data()

values_count = numpy.zeros(20)
controversial_count = numpy.zeros(20)
for i in range(0, len(cont_scores)):
    values_count = numpy.add(values_count, preds[i])
    if int(cont_scores[i]) == 1:
        controversial_count = numpy.add(controversial_count, preds[i])

plot_1(values_count, controversial_count)
plot_4(values_count, controversial_count)