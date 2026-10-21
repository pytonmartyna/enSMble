import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def read_data(file_path: str, target: str, print_head: bool = False):
    """
        This function allows to read the input data from file of '.csv' or '.xlsx' format.

        Args:
            file_path (str): Path to the file with data.
            target (str): Name of the column containing class labels for each of observations in the dataset.
            print_head (bool): An argument determining whether to print first five records of result DataFrame or not.

        Returns:
            df_features (pd.DataFrame): DataFrame with set of features.
            target_col (pd.Series): Vector with class labels for each observation in the dataset.
    """
    try:
        if file_path.lower().endswith("xlsx"):
            df = pd.read_excel(file_path, engine='openpyxl')
            if target not in df.columns:
                raise ValueError("Kolumna " + target + " nie istnieje w zbiorze danych.")
            df_features = df.iloc[:, 2:]
            target_col = df[target]
            if print_head:
                print(df.head())
            return df_features, target_col
        elif file_path.lower().endswith("csv"):
            df = pd.read_csv(file_path)
            if target not in df.columns:
                raise ValueError("Kolumna " + target + " nie istnieje w zbiorze danych.")
            df_features = df.iloc[:, 2:]
            target_col = df[target]
            if print_head:
                print(df.head())
            return df_features, target_col
        else:
            raise ValueError("Nieobsługiwany format pliku.")
    except FileNotFoundError:
        print("Plik " + file_path + " nie istnieje.")
                

def preprocess(data: pd.DataFrame,
               target_col: pd.Series=None,
               deal_with_nan_val: bool = True,
               scale_data: bool = True,
               code_target_: bool = False):
    """
        This function performs data preprocessing including scaling, imputation of missing data, etc.

        Args:
             data (pd.DataFrame): Dataset to be preprocessed.
             deal_with_nan_val (bool): An argument determining whether the function should impute missing data or not.
             scale_data (bool): An argument determining whether the function should scale data or not.
             code_target_ (bool): An argument determining whether the function should code values
             contained in target vector or not (applicable in case of categorical class labels).

        Returns:
            preprocessed_data (pd.DataFrame): Preprocessed dataset.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Given data is not pandas.DataFrame type, try again.")
    imputer = SimpleImputer()
    scaler = StandardScaler()
    preprocessed_data = data.copy()
    if deal_with_nan_val == True:
        preprocessed_data = deal_nan(preprocessed_data, imputer)
    if scale_data == True:
        preprocessed_data = scale(preprocessed_data, scaler)
    if code_target_ == True:
        if target_col is None or target_col.empty:
            raise ValueError("To perform target coding a vector with class labels is needed.")
        code_target_col = code_target(target_col)
        return preprocessed_data, code_target_col
    return preprocessed_data


def split(data: pd.DataFrame, target: pd.Series, test_size: float, random_state: int, shuffle: bool,
          stratify: pd.Series = None):
    """
        This function splits the input dataset into training and testing subsets.

        Args:
            data (pd.DataFrame): Input dataset to be split.
            target (pd.Series): Vector of class labels for input dataset.
            test_size (float): Size of the test subset.
            random_state (int): Random generator seed to ensure repeatable results.
            shuffle (bool): An argument determining whether observations contained in the input dataset should be
            shuffled before splitting into subsets.
            stratify (pd.Series): Vector of class labels for the input dataset according to which data will be stratified.

        Returns:
            data_train (pd.DataFrame): Training subset.
            data_test (pd.DataFrame): Testing subset.
            target_train (pd.Series): Vector of class labels for training subset.
            target_test (pd.Series): Vector of class labels for testing subset.
    """
    if not isinstance(test_size, float) or not (0 < test_size < 1):
        raise ValueError("Test size must be a float and belong to the interval (0,1). Try again.")
    data_train, data_test, target_train, target_test = train_test_split(
        data, target,
        test_size = test_size,
        random_state = random_state,
        shuffle = shuffle,
        stratify = target if isinstance(stratify, pd.Series) and len(target) == len(data) else None)
    return data_train, data_test, target_train, target_test


def deal_nan(data: pd.DataFrame, imputer: SimpleImputer):
    """
        This function performs in-place imputation of missing data.

        Args:
            data (pd.DataFrame): Input dataset for which the missing data will be imputed.
            imputer (SimpleImputer): Imputer object to transform data and impute the missing values.

        Returns:
            data (pd.DataFrame): Dataset with imputed missing data.
    """
    for column in data.columns:
        missing_percentage = data[column].isna().sum() / len(data)
        if missing_percentage <= 0.1:
            data[column] = imputer.fit_transform(data[[column]])
        else:
            data.drop(columns = [column], inplace = True)
    return data


def scale(data: pd.DataFrame, scaler: StandardScaler):
    """
        This function performs data scaling.

        Args:
            data (pd.DataFrame): Input dataset to be scaled.
            scaler (StandardScaler): Scaler object to transform data and scale it.

        Returns:
            scaled_data (pd.DataFrame): Scaled dataset.
    """
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns = data.columns)

def code_target(target_col: pd.Series):
    """
        This function performs target values (class labels) coding.

        Args:
            target_col (pd.Series): Complete vector of class labels.

        Returns:
            coded_target_col (pd.Series): Complete vector of class labels with coded (numerical) values.
    """
    unique_vals = target_col.unique()
    indexes = enumerate(unique_vals)
    class_mapper = {target : index for index, target in indexes}
    coded_target_col = target_col.map(class_mapper)
    return coded_target_col


