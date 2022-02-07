from initial_regression_table import get_angreb_and_anerkendelse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_plt_for_latex():
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    plt.style.use('science')


def make_plots(dep_var: str, df_angreb: pd.DataFrame, df_anerkendelse: pd.DataFrame, subplots=False):
    fig, axs = plt.subplots(2, 2, dpi=120, figsize=(10, 6))

    if dep_var == "like_count_x":
        dep_string = "likes"
    elif dep_var == "share_count":
        dep_string = "delinger"

    if subplots:
        one_p_max = df_angreb[dep_var].sort_values(ascending=False).values[round(len(df_angreb) * 0.01)]
        df_angreb = df_angreb.loc[df_angreb[dep_var] <= one_p_max]

        one_p_max = df_anerkendelse[dep_var].sort_values(ascending=False).values[round(len(df_anerkendelse) * 0.01)]
        df_anerkendelse = df_anerkendelse.loc[df_anerkendelse[dep_var] <= one_p_max]

    x, y = df_angreb["angreb_x"], df_angreb[dep_var]
    m, b = np.polyfit(x, y, 1)
    axs[0][0].plot(x, y, 'o', alpha=0.05)
    axs[0][0].plot(x, m * x + b, color="k")
    axs[0][0].set_ylabel(f'{dep_string}')
    axs[0][0].set_title(f'Angreb-niveau og {dep_string}')

    axs[1][0].plot(x, y, 'o', alpha=0.05)
    axs[1][0].plot(x, m * x + b, color="k")
    axs[1][0].set_yscale("log")
    axs[1][0].set_ylabel(f'log({dep_string})')
    axs[1][0].set_xlabel("Aggregeret angreb-niveau for kommentarspor")

    x, y = df_anerkendelse["anerkendelse"], df_anerkendelse[dep_var]
    m, b = np.polyfit(x, y, 1)
    axs[0][1].plot(x, y, 'o', alpha=0.05)
    axs[0][1].plot(x, m * x + b, color="k")
    axs[0][1].set_ylabel(f'{dep_string}')
    axs[0][1].set_title(f'Anerkendelses-niveau og {dep_string}')

    axs[1][1].plot(x, y, 'o', alpha=0.05)
    axs[1][1].plot(x, m * x + b, color="k")
    axs[1][1].set_yscale("log")
    axs[1][1].set_ylabel(f'log({dep_string})')
    axs[1][1].set_xlabel("Aggregeret anerkendelses-niveau for kommentarspor")

    fig.tight_layout()
    if subplots:
        fig.savefig(f'../../figures/presentation_08_02/lineplot_{dep_string}_subset')
    if not subplots:
        fig.savefig(f'../../figures/presentation_08_02/lineplot_{dep_string}')


if __name__ == "__main__":
    df_angreb, df_anerkendelse = get_angreb_and_anerkendelse()
    set_plt_for_latex()
    for var in ["like_count_x", "share_count"]:
        for bool in [False, True]:
            make_plots(dep_var=var, df_angreb=df_angreb, df_anerkendelse=df_anerkendelse, subplots=bool)
