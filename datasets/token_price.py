import itertools
from typing import Tuple
from giza_datasets import DatasetsLoader
import polars as pl
from datasets.utils import delete_null_columns, normalize

class ApyDataset:

    @staticmethod
    def load(token_name: str, starter_date: str, loader: DatasetsLoader) -> pl.DataFrame:
        """
        Manipulates the APY dataset for a specific token to prepare it for analysis.

        Returns:
        A pivoted DataFrame focused on the specified token, with each row representing a date and each column representing a different project's TVL and APY.
        """
        apy_df = loader \
            .load("top-pools-apy-per-protocol") \
            .filter(pl.col("underlying_token").str.contains(token_name)) \
            .with_columns(
                    pl.col("project") + "_" + pl.col("chain") +  pl.col("underlying_token")
            ) \
            .drop(["underlying_token", "chain"])

        unique_projects = apy_df \
            .filter(pl.col("date") <= starter_date) \
            .select("project") \
            .unique()

        apy_df_token = apy_df.join(
            unique_projects, 
            on="project", 
            how="inner"
        )

        return apy_df_token.pivot(
            index="date",
            columns="project",
            values=["tvlUsd", "apy"]
        )

class TLVDataset:

    @staticmethod
    def load(token_name: str, starter_date: str, loader: DatasetsLoader) -> pl.DataFrame:
        """
        Manipulates the TVL (Total Value Locked) dataset for a specific token to prepare it for analysis.

        Returns:
        A pivoted DataFrame focused on the specified token's TVL, with each row representing a date and each column representing a different project's TVL.
        """
        tvl_df = loader \
            .load("tvl-per-project-tokens") \
            .unique(subset=["date", "project"]) \
            .filter(pl.col("date") > starter_date) \

        return tvl_df[[token_name, "project", "date"]].pivot(
            index="date",
            columns="project",
            values=token_name
        )


class PriceDataset:

    @staticmethod
    def load(token_name: str, target_lag: int, loader: DatasetsLoader) -> pl.DataFrame:
        """
        Loads and processes the main dataset for a specific token, generating features and calculating correlations.

        Returns:
        A DataFrame ready for further analysis or model training, containing combined and processed features from all datasets.
        """
        daily_token_prices = loader.load("tokens-daily-prices-mcap-volume")
        df_main = PriceDataset.__get_df_main(token_name, target_lag, daily_token_prices)
        df_main = PriceDataset.__get_lag_correlations(token_name, df_main, daily_token_prices)
        return df_main

    @staticmethod
    def __get_df_main(token_name: str, target_lag: int, daily_token_prices: pl.DataFrame) -> pl.DataFrame:
        """
        Performs the main dataset manipulation including loading the dataset, generating features, and calculating correlations.

        This function executes several steps:
        - Loads the 'tokens-daily-prices-mcap-volume' dataset.
        - Filters the dataset for a specified token.
        - Calculates various features such as price differences and trends over different days.
        - Adds day of the week, month of the year, and year as features.
        - Calculates lag correlations between the specified token and all other tokens in the dataset.
        - Identifies the top 10 correlated tokens based on 15-day lag correlation.
        - Joins features of the top 10 correlated tokens to the main dataframe.
        
        Returns:
        A DataFrame containing the original features along with the newly calculated features and correlations for further analysis.
        """

        return daily_token_prices \
            .filter(pl.col("token") == token_name) \
            .with_columns(
                ((pl.col("price").shift(-target_lag) - pl.col("price")) > 0).cast(pl.Int8).alias("target")
            ) \
            .with_columns(
                PriceDataset.__get_column_aliases(token_name)
            ) \
            .with_columns([
                pl.col("date").dt.weekday().alias("day_of_week"),
                pl.col("date").dt.month().alias("month_of_year"),
                pl.col("date").dt.year().alias("year")
            ])
        
    @staticmethod
    def __calculate_lag_correlations(df, lags=[1, 3, 7, 15]):
        """
        Calculates and returns the lagged correlations between different tokens' prices in the dataset.
        """
        correlations = {}
        tokens = df.select("token").unique().to_numpy().flatten()
        for base_token in tokens:
            for compare_token in tokens:
                if base_token == compare_token:
                    continue
                base_df = df.filter(pl.col("token") == base_token).select(["date", "price"]).sort("date")
                compare_df = df.filter(pl.col("token") == compare_token).select(["date", "price"]).sort("date")
                merged_df = base_df.join(compare_df, on="date", suffix="_compare")
                key = f"{base_token}_vs_{compare_token}"
                correlations[key] = {}
                for lag in lags:
                    merged_df_lagged = merged_df.with_columns(pl.col("price_compare").shift(lag))
                    corr_df = merged_df_lagged.select(
                        pl.corr("price", "price_compare").alias("correlation")
                    )
                    corr = corr_df.get_column("correlation")[0]
                    correlations[key][f"lag_{lag}_days"] = corr
                    
        return correlations
        
    @staticmethod
    def __get_lag_correlations(token_name: str, df_main: pl.DataFrame, daily_token_prices: pl.DataFrame) -> pl.DataFrame:


        correlations = PriceDataset.__calculate_lag_correlations(daily_token_prices)

        data = []
        for tokens, lags in correlations.items():
            base_token, compare_token = tokens.split('_vs_')
            for lag, corr_value in lags.items():
                data.append({'Base Token': base_token, 'Compare Token': compare_token, 'Lag': lag, 'Correlation': corr_value})

        df_correlations = pl.DataFrame(data)

        top_10_correlated_coins = df_correlations.filter(
            (pl.col("Base Token") == token_name) & \
            (pl.col("Lag") == "lag_15_days")
        ) \
        .sort(by="Correlation", descending = True)["Compare Token"].to_list()[0:10]
        

        for token in top_10_correlated_coins:
            df_token = daily_token_prices \
                .filter(pl.col("token") == token)
            
            df_token_features = df_token \
                .with_columns(
                    PriceDataset.__get_column_aliases(token)
                ).select(
                    [pl.col("date")] + \
                    list(map(PriceDataset.__price_diff_tag, [1, 3, 7, 15], 4 * [token])) + \
                    list(map(PriceDataset.__trend_tag, [1, 3, 7, 15], 4 * [token]))
                )

            return df_main.join(df_token_features, on="date", how="left")
        
    @staticmethod
    def __price_diff_tag(days, token):
        return f"diff_price_{days}_days_ago{token}"
    
    @staticmethod
    def __trend_tag(days, token):
        return f"trend_{days}_day{token}"

    @staticmethod
    def __get_column_aliases(token: str) -> Tuple[str, str]:
        relevant_days = [1, 3, 7, 15, 30]

        return list(itertools.chain(*[
            (
                (pl.col("price").diff(n = days).alias(PriceDataset.__price_diff_tag(days, token))),
                ((pl.col("price") - pl.col("price").shift(days)) > 0).cast(pl.Int8).alias(PriceDataset.__trend_tag(days, token))
            ) for days in relevant_days
        ]))

class TokenDataset:

    @staticmethod
    def get_train_test_split(
        token_name: str,
        starter_date: str,
        target_lag: int,
        split_ratio: float=0.85,
        loader: DatasetsLoader=DatasetsLoader()
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    
        """
        Prepares the datasets for training and testing by selecting features, splitting the data, and preprocessing.

        Parameters:
        - df: The main DataFrame containing features and targets.

        Returns:
        Prepared feature and target DataFrames for both training and testing.
        """
        df = TokenDataset.load(token_name, starter_date, target_lag, loader)
        
        # We will use all columns except the date, month_of_year, price, and target columns as features.
        features = list(set(df.columns) - set(["date","month_of_year", "price", "target"]))

        # Split the data into training and testing sets.
        cutoff_index = int(len(df) * split_ratio)
        df_train, df_test = df[:cutoff_index], df[cutoff_index:]

        # Normalize the features.
        x_train, x_test = normalize(df_train[features]), normalize(df_test[features])

        # Get the evaluation labels, which are the target values.
        y_train, y_test = df_train.select('target'), df_test.select('target')

        return x_train, x_test, y_train, y_test


    @staticmethod
    def load(token_name: str, starter_date: str,  target_lag: int, loader: DatasetsLoader) -> pl.DataFrame:
        """
        Loads and processes the main, APY, and TVL datasets, joining them on the date column and performing postprocessing.

        Returns:
        A DataFrame ready for further analysis or model training, containing combined and processed features from all datasets.
        """
        
        df_main = PriceDataset.load(token_name, target_lag, loader)
        apy_df = ApyDataset.load(token_name, starter_date, loader)
        tvl_df = TLVDataset.load(token_name, starter_date, loader)

        df_main = df_main.join(tvl_df, on = "date", how = "inner")
        df_main = df_main.join(apy_df, on = "date", how = "inner")

        num_rows_to_select = len(df_main) - target_lag
        df_main = df_main.slice(0, num_rows_to_select)

        # Some of the extra tokens we added do not have much historical information, so we raised the minimum date of our dataset a little bit.
        df_main = df_main.filter(pl.col("year") >= 2022)
        df_main = df_main.drop(["token","market_cap"])
        df_main = delete_null_columns(df_main, 0.2)
        return df_main