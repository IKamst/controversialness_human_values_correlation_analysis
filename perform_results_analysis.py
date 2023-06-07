import pandas as pd
import re
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


VALUES = ["Self-direction: thought",
          "Self-direction: action",
          "Stimulation", "Hedonism",
          "Achievement", "Power: dominance",
          "Power: resources", "Face",
          "Security: personal", "Security: societal",
          "Tradition", "Conformity: rules",
          "Conformity: interpersonal", "Humility",
          "Benevolence: caring", "Benevolence: dependability",
          "Universalism: concern", "Universalism: nature",
          "Universalism: tolerance", "Universalism: objectivity"]


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
    data_without_deleted = data_without_nans[data_without_nans['comment'].map(contains_deleted) != True]
    clean_data = data_without_deleted[data_without_deleted['comment'].map(contains_removed) != True]

    # Reset indexes after deleting rows:
    clean_data.reset_index(inplace=True)

    cont_scores = [int(score) for score in clean_data['cont']]
    preds = list(map(str_to_list, clean_data['values']))
    comments_list = [str(comment) for comment in clean_data['comment']]

    return cont_scores, preds, comments_list


def make_word_clouds(comments_list):
    non_con_comm_list = []
    con_comm_list = []
    # Put all controversial comments in 1 list, and non-controversial in another:
    for idx, val in enumerate(cont_scores):
        if val == 0:
            non_con_comm_list.append(comments_list[idx])
        if val == 1:
            con_comm_list.append(comments_list[idx])

    # Make long strings out of the (non-)controversial lists:
    non_con_comm = " ".join(word for word in non_con_comm_list)
    con_comm = " ".join(word for word in con_comm_list)

    stopwords = ["gt"] + list(STOPWORDS)
    wordcloud_non_con_comm = WordCloud(width=800, height=800,
                                       background_color='white',
                                       stopwords=stopwords,
                                       min_font_size=10).generate(non_con_comm)

    wordcloud_con_comm = WordCloud(width=800, height=800,
                                   background_color='white',
                                   stopwords=stopwords,
                                   min_font_size=10).generate(con_comm)

    # Plot the wordclouds side-by-side:
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(wordcloud_con_comm)
    ax2.imshow(wordcloud_non_con_comm)
    plt.tight_layout(pad=0)
    plt.show()


def perform_logit_reg(cont_scores, preds):
    # Check the coefficients from logistic regression:
    clf = LogisticRegression(random_state=0).fit(preds, cont_scores)
    logit_coefs = pd.DataFrame(columns=['value', 'coef'])
    for value_idx, value in enumerate(VALUES):
        print(f"{value}: {clf.coef_[0][value_idx]}")
        logit_coefs.loc[value_idx] = [value] + [clf.coef_[0][value_idx]]
    plt.bar(logit_coefs.value, logit_coefs.coef)
    plt.xlabel("Human values")
    plt.ylabel("Coefficient")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def compute_jaccard_distance(preds):
    for value_idx, val in enumerate(VALUES):
        curr_value = [item[value_idx] for item in preds]

        for value_idx_0, val_0 in enumerate(VALUES):
            # No need to compute d_J(x,x):
            if value_idx_0 == value_idx:
                continue

        curr_value_0 = [item[value_idx_0] for item in preds]
        if distance.jaccard(curr_value, curr_value_0) < 0.65 and val != val_0:
            print(f"{val} AND {val_0}: "f"{distance.jaccard(curr_value, curr_value_0)}")


cont_scores, preds, comments_list = read_data()
make_word_clouds(comments_list)
perform_logit_reg(cont_scores, preds)
compute_jaccard_distance(preds)
