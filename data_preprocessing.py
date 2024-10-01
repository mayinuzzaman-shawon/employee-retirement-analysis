"""
Data preprocessing and DataLoader preparation module.

Provides:
- `balance_classes`: Downsamples the majority class in a dataset.
- `load_data`: Loads and preprocesses data from a CSV file with options for balancing and imputation.
- `get_data_loaders`: Creates PyTorch DataLoaders for training and testing.

Dependencies:
- pandas, torch, sklearn, plotting (internal module)
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from plotting import plot_class_distribution

def balance_classes(data: pd.DataFrame, target_column: str, majority_ratio: float, random_seed: int) -> pd.DataFrame:
    """
    Balance the classes in the dataset by downsampling the majority class.

    Args:
        data (pd.DataFrame): The dataset containing the target column to be balanced.
        target_column (str): The name of the target column in the dataset.
        majority_ratio (float): The ratio of the majority class samples to retain compared to its original size.
        random_seed (int): Seed for random number generation to ensure reproducibility.

    Returns:
        pd.DataFrame: A new DataFrame with balanced classes, where the majority class has been downsampled 
                      according to the specified `majority_ratio`.
    """
    # Determine the majority and minority classes
    class_counts = data[target_column].value_counts()
    majority_class = class_counts.idxmax()  # class with the most samples
    minority_class = class_counts.idxmin()  # class with the fewest samples

    # Calculate the number of samples to keep for the majority class
    majority_count_to_keep = int(class_counts[majority_class] * majority_ratio)
    print("Balancing Dataaset ...") 
    print("Majority Class:", int(majority_class)) 
    print("Minority Class:", int(minority_class))
    print("Majority Samples To Keep:", majority_count_to_keep) 

    # Downsample the majority class
    df_majority_downsampled = resample(data[data[target_column] == int(majority_class)], 
                                       replace=False,    
                                       n_samples=majority_count_to_keep, 
                                       random_state=random_seed)  # reproducible results

    # Combine the minority class with the downsampled majority class
    balanced_data = pd.concat([df_majority_downsampled, data[data[target_column] == minority_class]])

    # Shuffle the dataset
    balanced_data = balanced_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return balanced_data

def load_data(
        data_path: str, 
        selected_columns:list[str]=None, 
        balance:bool=False, 
        balance_majority_ratio:float=0.7, 
        frac:float=None, 
        impute_missing_values: bool=False,
        random_seed:int=42
    ):
    """
    Load, preprocess, and optionally balance data from a CSV file.

    Args:
        data_path (str): Path to the CSV file containing the dataset.
        selected_columns (list[str], optional): List of columns to select from the dataset. If None, a predefined set of 
                                                columns is used. Defaults to None.
        balance (bool, optional): Whether to balance the classes in the 'pemlr' target variable. Defaults to False.
        balance_majority_ratio (float, optional): The ratio of the majority class to retain when balancing. 
                                                  Applicable only if `balance` is True. Defaults to 0.7.
        frac (float, optional): Fraction of the data to sample. If None, the entire dataset is used. Defaults to None.
        impute_missing_values (bool, optional): Whether to impute missing values using the median. If False, rows with 
                                                missing values are dropped. Defaults to False.
        random_seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the feature matrix (X) and the target variable (y).
    """
    print("Loading Dataset ...")
    
    # Default columns if none are provided
    default_columns = [
        "hrhhid2", "hufinal", "hubus", "gereg", "gestfips", "hrnumhou",
        "hefaminc", "pesex", "pemaritl", "peeduca", "peafever", "penatvty",
        "pecyc", "prdasian", "ptdtrace", "prdthsp", "prnmchld", "prchld",
        "prcitflg", "prcitshp", "prtage", "prtfage", "pehrwant", "pemlr",
        "prdisflg", "ptio1ocd", "ptio2ocd", "hrhhid"
    ]

    # Load the data
    data = pd.read_csv(data_path)
    
    # Use selected_columns if provided, otherwise use default_columns
    if selected_columns is None:
        selected_columns = default_columns
    
    # Ensure that all selected_columns are in the dataframe
    missing_columns = [col for col in selected_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing in the data: {missing_columns}")
    
    data = data[selected_columns]
    
    # Drop rows with missing values
    if not impute_missing_values:
        data = data.dropna()
    
    # Transform 'pemlr' to binary classification (1: retired, 0: not retired)
    data = data[data['pemlr'] != -1]

    # converting 'pemlr' to a binary target variable where 1 indicates retired (pemlr=5) and 0 indicates not retired (pemlr!=5)
    data['pemlr'] = data['pemlr'].apply(lambda x: 1 if x == 5 else 0)
    
    # Sample the data if frac is provided
    if frac is not None:
        data = data.sample(frac=frac, random_state=random_seed)  # `random_state` ensures reproducibility
    
    # Balance the classes if balance is True
    plot_title = "Dataset Class Distribution"
    plot_filename = "artifacts/dataset_class_distribution.png"
    if balance:
        data = balance_classes(data, target_column='pemlr', majority_ratio=balance_majority_ratio, random_seed=random_seed)
        plot_title = "Balanced Dataset Class Distribution"
        plot_filename = "artifacts/balanced_dataset_class_distribution.png"
    else:
        data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Separate features and target
    X = data.drop('pemlr', axis=1)
    X = pd.get_dummies(X, drop_first=True)

    if impute_missing_values:
        imputer = SimpleImputer(strategy='median')  # Impute missing values with the median value
        X = imputer.fit_transform(X)
    
    y = data['pemlr']
    
    # Plot class distribution
    plot_class_distribution(y, title=plot_title, save_path=plot_filename)

    return X, y

def get_train_test_data(data_path, frac=None, balance:bool=False, balance_majority_ratio=0.3, random_seed=42):
    X, y = load_data(
        data_path, 
        frac=frac,
        balance=balance, 
        balance_majority_ratio=balance_majority_ratio,
    )
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def get_data_loaders(
    data_path: str,
    test_size: float = 0.2,
    batch_size: int = 1024,
    selected_columns:list[str]=None, 
    balance:bool=False,
    balance_majority_ratio=0.7,
    random_seed:int=42,
    device: torch.device = torch.device('cpu')
):
    """
    Prepare DataLoaders for training and testing from a CSV file.

    Args:
        data_path (str): Path to the CSV file containing the dataset.
        test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.2.
        batch_size (int): Number of samples per batch to load. Defaults to 1024.
        selected_columns (list[str], optional): List of column names to use as features. If None, all columns are used.
        balance (bool): Whether to balance the dataset by oversampling the minority class. Defaults to False.
        balance_majority_ratio (float): The ratio of the majority class size to the minority class size after balancing.
                                        Only applicable if `balance` is True. Defaults to 0.7.
        random_seed (int): Seed for random number generation, ensuring reproducibility. Defaults to 42.
        device (torch.device): The device on which tensors should be loaded. Defaults to CPU.

    Returns:
        Tuple[DataLoader, DataLoader, int]: A tuple containing the training DataLoader, testing DataLoader, 
                                            and the input dimension size (number of features).
    """
    
    X, y = load_data(
        data_path=data_path, 
        selected_columns=selected_columns, 
        balance=balance, 
        balance_majority_ratio=balance_majority_ratio, 
        frac=None, 
        random_seed=random_seed
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = X_train_tensor.shape[1]
    
    return train_loader, test_loader, input_size
