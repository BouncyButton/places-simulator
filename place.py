import simpy
import numpy as np

AVG_DINING_TIME = 90


class Place:
    went_away: int

    def __init__(self, env, name='Unnamed place', available_seats=50, tables4=5, tables6=3, tables8=3):
        self.reception = simpy.Resource(env, capacity=1)
        self.name = name
        self.__available_seats = available_seats
        # self.full = env.event()
        self.when_full = None
        self.went_away = 0
        self.tables4 = simpy.Resource(env, capacity=tables4)
        self.tables6 = simpy.Resource(env, capacity=tables6)
        self.tables8 = simpy.Resource(env, capacity=tables8)

        # todo un sacco

    @property
    def available_seats(self):
        return (self.tables4.capacity - self.tables4.count) * 4 + (self.tables6.capacity - self.tables6.count) * 6 + (
                self.tables8.capacity - self.tables8.count) * 8

    def total_seats(self):
        return self.tables4.capacity * 4 + self.tables6.capacity * 6 + self.tables8.capacity * 8

    @available_seats.setter
    def available_seats(self, v):
        self.__available_seats = v

    def get_random_dining_time(self):
        return np.random.lognormal(mean=0.0, sigma=0.25) * AVG_DINING_TIME

    def get_affluence_status(self):
        percentage = (1 - self.available_seats / self.total_seats())
        if percentage > 0.9:
            return 4
        elif percentage > 0.7:
            return 3
        elif percentage > 0.5:
            return 2
        else:
            return 1

