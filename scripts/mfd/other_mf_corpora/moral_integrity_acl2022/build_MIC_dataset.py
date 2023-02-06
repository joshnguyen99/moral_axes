import pandas as pd
import sys
# read mic data
mic_path = './data/MIC.csv'
mic = pd.read_csv(mic_path)

# create test path for all datasets
mic_test_path = './data/MIC_dataset.csv'

# create 5 foundation columns
foundations = ['care', 'fairness', 'authority', 'loyalty', 'sanctity']


# to process social chemistry & mic data
def convert_rot_to_test(dataframe, columns, path):
    for foundation in foundations:
        dataframe[foundation] = 0
        dataframe.loc[dataframe[columns].notnull() & dataframe[columns].str.contains(foundation), foundation] = 1
    dataframe_test = dataframe[dataframe['split'] == 'dev'][[
        'rot', 'care', 'fairness', 'loyalty', 'authority', 'sanctity']]

    dataframe_test.to_csv(path)


def main():
    # convert to the test format: text,'care','fairness','authority','loyalty','sanctity'
    convert_rot_to_test(mic, 'moral', mic_test_path)


if __name__ == "__main__":
    main()
