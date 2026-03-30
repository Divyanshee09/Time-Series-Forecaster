import pandas as pd
for name in ['electricity.csv','m4_hourly.csv','airline.csv','bitcoin.csv','covid_cases.csv']:
    df = pd.read_csv(f'data/real/{name}')
    df['ds'] = pd.to_datetime(df['ds'])
    print(f"{name}: N={len(df)}, y in [{df.y.min():.1f}, {df.y.max():.1f}], "
          f"dates {df.ds.iloc[0].date()} to {df.ds.iloc[-1].date()}")
