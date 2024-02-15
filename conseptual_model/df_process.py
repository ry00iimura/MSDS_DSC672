# load libraries
import pandas as pd

class D:

    """
    A class to manage and manipulate data loaded from a CSV file.

    Attributes:
    -----------
    df : DataFrame
        A pandas DataFrame containing the data loaded from the specified CSV file.

    Methods:
    --------
    del_white_space_col(inplace):
        Renames DataFrame columns by removing leading and trailing whitespace.

    get_questions_as_df():
        Returns a DataFrame containing the column names (excluding the first column) as questions.
    """
     
    def __init__(self,file_path,encode):

        """
        Initializes the D object by loading data from a CSV file into a pandas DataFrame.

        Parameters:
        -----------
        file_path : str
            The file path to the CSV file to be loaded.
        encode : str
            The encoding of the CSV file.
        """

        self.df = pd.read_csv(file_path,encoding = encode)

    def del_white_space_col(self, inplace):

        """
        Renames DataFrame columns by removing leading and trailing whitespace.

        Parameters:
        -----------
        inplace : bool
            If True, modifies the DataFrame in place. Otherwise, returns a modified copy.

        Returns:
        --------
        DataFrame or None
            Returns the modified DataFrame if inplace is False. Otherwise, modifies the DataFrame in place and returns None.
        """

        if inplace == True:

            self.df = self.df.rename(columns = dict(zip(self.df.columns,[c.strip() for c in self.df.columns])))

            return self.df

        else:
    
            return self.df.rename(columns = dict(zip(self.df.columns,[c.strip() for c in self.df.columns])))
    
    def get_questions_as_df(self):

        """
        Returns a DataFrame containing the column names (excluding the first column) as questions.

        Returns:
        --------
        DataFrame
            A DataFrame where each row represents a question (column name) from the original DataFrame, excluding the first column.
        """

        return pd.DataFrame(self.df.columns[1:], index = [i for i in range(len(self.df.columns[1:]))],columns = ['question'])
