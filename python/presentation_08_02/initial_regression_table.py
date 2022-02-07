import pandas as pd
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer


def get_angreb_and_anerkendelse(df: pd.DataFrame = pd.read_pickle("/Users/frederikskovlundkilpinen/Documents/Social Data Science/Thesis/Data/df_post_level.pkl")):
    df_angreb = df[~pd.isna(df["angreb_x"])][["id_x", "angreb_x", "like_count_x", "share_count"]]
    df_anerkendelse = df[~pd.isna(df["anerkendelse"])][["id_x", "anerkendelse", "like_count_x", "share_count"]]
    return df_angreb, df_anerkendelse


def get_params(df_angreb: pd.DataFrame, df_anerkendelse: pd.DataFrame, subset = False):
    if subset:
        one_p_max = df_angreb["like_count_x"].sort_values(ascending=False).values[round(len(df_angreb) * 0.01)]
        df_angreb = df_angreb.loc[df_angreb["like_count_x"] <= one_p_max]

        one_p_max = df_anerkendelse["share_count"].sort_values(ascending=False).values[round(len(df_anerkendelse) * 0.01)]
        df_anerkendelse = df_anerkendelse.loc[df_anerkendelse["share_count"] <= one_p_max]

    angreb_likes = smf.ols(formula='like_count_x ~ angreb_x', data=df_angreb).fit()
    angreb_shares = smf.ols(formula='share_count ~ angreb_x', data=df_angreb).fit()
    anerkendelse_likes = smf.ols(formula='like_count_x ~ anerkendelse', data=df_anerkendelse).fit()
    anerkendelse_shares = smf.ols(formula='share_count ~ anerkendelse', data=df_anerkendelse).fit()
    return Stargazer([angreb_likes, angreb_shares, anerkendelse_likes, anerkendelse_shares])


def modify_table(table: Stargazer):
    table.covariate_order(["angreb_x", "anerkendelse", "Intercept"])
    table.rename_covariates({"angreb_x": "angreb (0-1, kontinuert)",
                             "anerkendelse": "anerkendelse (0-1, kontinuert)",
                             "Intercept": "Konstant"})
    table.show_model_numbers(False)
    table.show_degrees_of_freedom(False)
    return table


if __name__ == "__main__":
    df_angreb, df_anerkendelse = get_angreb_and_anerkendelse()
    for bool in [False, True]:
        regression_table = get_params(df_angreb, df_anerkendelse, subset=bool)
        regression_table = modify_table(regression_table)
        print(regression_table.render_latex())
