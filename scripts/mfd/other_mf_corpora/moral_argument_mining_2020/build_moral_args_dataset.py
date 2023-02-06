import pandas as pd
import sys

# read 2 moral args and argquality corpus
moral_args_1_path = './data/dagstuhl_morality_1.csv'
moral_args_2_path = './data/dagstuhl_morality_2.csv'
args_quality_path = './data/argquality_corpus.csv'

moral_args_1 = pd.read_csv(moral_args_1_path,sep=';')
moral_args_2 = pd.read_csv(moral_args_2_path,sep=';')
args_quality = pd.read_csv(args_quality_path,sep='\t', encoding='latin-1')
args_quality = args_quality.set_index('#id')

# to join 2 annotators' labelled datasets: rename columns 'MF1', 'MF2' of moral_args_2 to join moral_args_1 
moral_args_1[['MF4','MF5']] = moral_args_2[['MF1','MF2']]
# print(moral_args_1.head())

# create dataset path for all datasets
moral_args_dataset_path = './data/moral_args_dataset.csv'


# create 5 foundation columns
foundations = ['care','fairness','authority','loyalty','purity']


# to process and join moral arguments data
def convert_argument_to_test(dataframe,path):
    
    for foundation in foundations:
        dataframe[foundation]=0
        dataframe.loc[(dataframe['MF1']==foundation)|(dataframe['MF2']==foundation)|(dataframe['MF3']==foundation)|(dataframe['MF4']==foundation)|(dataframe['MF5']==foundation),foundation]=1



    dataframe = dataframe[['#id','care','fairness','loyalty','authority','purity']]
    dataframe = dataframe.rename(columns={'purity':'sanctity'})
    
    # set '#id' to be tyhe index and jin the corpus with ArgQuality corpus
    dataframe = dataframe.set_index('#id')
    dataframe = dataframe.join(args_quality['argument'],on='#id').drop_duplicates()
    dataframe.to_csv(path)

def main():
    # convert to the test format: text,'care','fairness','authority','loyalty','sanctity'
    convert_argument_to_test(moral_args_1,moral_args_dataset_path)


if __name__ == "__main__":
    main()