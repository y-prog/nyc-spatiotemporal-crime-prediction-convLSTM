from data_cleaning_methods import NYCCrimePreprocessor as Preprocessor

def main(config_yaml_file):
    preprocessor = Preprocessor(config_yaml_file)
    df = preprocessor.data_to_pandas_df()
    df_cleaned = preprocessor.select_and_clean_spacetime_columns(df)
    df_bounded = preprocessor.remove_outer_coordinates(df_cleaned)
    df_in_grid = preprocessor.df_included_grid(df_bounded)
    tensor = preprocessor.generate_tensor(df_in_grid)
    return df_in_grid, tensor