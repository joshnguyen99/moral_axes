import pandas as pd
import sys
# read social chem data
social_chem_path = './data/social-chem-101.tsv'

social_chem = pd.read_csv(social_chem_path, sep='\t')

# create test path for all datasets
social_chem_test_path = './data/social_chem_test.csv'

# create 5 foundation columns
foundations = ['care','fairness','authority','loyalty','sanctity']


# to process social chemistry & mic data
def convert_rot_to_test(dataframe,columns,path):
    for foundation in foundations:
        dataframe[foundation]=0
        dataframe.loc[dataframe[columns].notnull() & dataframe[columns].str.contains(foundation),foundation]=1
    dataframe_test = dataframe[dataframe['split']=='dev'][['rot','care','fairness','loyalty','authority','sanctity']]

    dataframe_test.to_csv(path)

def main():
    # convert to the test format: text,'care','fairness','authority','loyalty','sanctity'
    convert_rot_to_test(social_chem,'rot-moral-foundations',social_chem_test_path)

if __name__ == "__main__":
    main()