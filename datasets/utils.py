import polars as pl

def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalizes the columns of a DataFrame using the Z-score standardization method.
    Replaces null values with the mean of the column.
    """
    for col in df.columns:
        mean_val = df[col].mean()
        std_dev = df[col].std() if df[col].std() != 0 else 1
        df = df.with_columns(((df[col].fill_null(mean_val) - mean_val) / std_dev).alias(col))
    return df

def delete_null_columns(df: pl.DataFrame, null_percentaje: float) -> pl.DataFrame:
    """    
    Removes columns from a dataframe where the percentage of null values exceeds a specified threshold.
    """
    threshold = df.shape[0] * null_percentaje
    columns_to_keep = [
        col_name for col_name in df.columns if df[col_name].null_count() <= threshold
    ]
    return df.select(columns_to_keep)