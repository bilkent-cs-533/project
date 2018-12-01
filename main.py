import csv

def main():
    train_tweets, train_labels = read_csv_file("train_tweets.csv")


def read_csv_file(csv_file_name):
    with open(csv_file_name, encoding = "utf-8") as f:
        csv_reader = csv.reader(f, delimiter=',')
        train_tweets = [line[0]  for line in csv_reader]
        train_labels = [line[1]  for line in csv_reader]
    return train_tweets, train_labels

def remove_stop_words(words, stop_words_file_name="stop_words.txt"):
  with open(stop_words_file_name, 'r', encoding="utf-8") as f:
    stop_words = f.read().lower().strip().split()

  return [x for x in words if x not in stop_words]


if __name__ == "__main__":
    main()