import pandas as pd


class SimulationLog:
    def __init__(self):
        self.list_cache = []
        self.data = pd.DataFrame()

    def add_data(self, d: dict):
        self.list_cache.append(d)

    def get_df(self) -> pd.DataFrame:
        self.data = self.data.append(pd.DataFrame.from_dict(self.list_cache))
        return self.data


# s = SimulationLog()
# s.add_data({'a': 1})
# s.add_data({'a': 2})
# x = s.get_df()

# assert x.mean()['a'] == 1.5
