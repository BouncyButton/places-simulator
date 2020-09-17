import random

import matplotlib.pyplot as plt
import numpy
import simpy
import person
from place import Place
# from logging import info, warning, error, debug, critical
import logging
import pandas as pd

# create logger with 'spam_application'
from simulationlog import SimulationLog


def logging_setup():
    logger = logging.getLogger('affluence_simulator')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('tuttecose.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


RANDOM_SEED = None
# RANDOM_SEED = 42
SIM_TIME = (24 + 3) * 60
AVG_PEOPLE_PER_HOUR = 5000


class Simulator:
    def __init__(self):
        self.places = []
        self.people = []
        self.env = simpy.Environment()
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
        self.placegoer_progressive = 1

        # piazza dei signori @ sat 1/8/2020
        # https://www.google.com/maps/place/Piazza+dei+Signori/@45.4081213,11.8720232,15.94z/data=!4m12!1m6!3m5!1s0x477edb53abc98ae1:0x74cabbff2bccd18b!2sGiardini+Giotto+PD!8m2!3d45.4125956!4d11.8795587!3m4!1s0x477eda4f92916c7b:0x86955ca5264bd0d8!8m2!3d45.4076909!4d11.8734071?hl=en
        self.popular_hours_saturday = {4: 0.02, 5: 0.00, 6: 0.00, 7: 0.02, 8: 0.08, 9: 0.20, 10: 0.37, 11: 0.45,
                                       12: 0.39, 13: 0.24, 14: 0.14, 15: 0.16, 16: 0.26, 17: 0.37, 18: 0.42, 19: 0.40,
                                       20: 0.41, 21: 0.54, 22: 0.80, 23: 0.96, 24: 0.86, 25: 0.55, 26: 0.25, 27: 0.08}

        # TODO
        self.popular_hours_sunday = {}
        self.popular_hours_weekday = {}

        self.group_n = {1: 10, 2: 20, 3: 15, 4: 20, 5: 15, 6: 15, 7: 5}
        self.AVG_GROUP_N = sum([(n * p) / 100 for n, p in zip(self.group_n.keys(), self.group_n.values())])

    def view(self):
        pass

    def get_available_seats(self):
        total = 0
        for p in self.places:
            total += p.available_seats
        return total

    def get_total_seats(self):
        total = 0
        for p in self.places:
            total += p.total_seats()
        return total


    def customer_arrivals(self, simlog):
        """Create new *placegoers* until the sim time reaches 120."""
        yield self.env.timeout(4 * 60)  # aspetto che siano le 4 di mattina
        logger.debug(" *** Comincia una nuova giornata")
        prevh = 0
        currh = -1

        try:
            while True:
                hour = int(self.env.now / 60)
                busy_factor = self.popular_hours_saturday[hour]

                if prevh != currh:
                    logger.debug(" ** Siamo le %d (busy=%d%%) (%d)" % (hour, busy_factor * 100, self.env.now))
                    currh = hour

                # logger.debug("busy_factor={0} AVG_PPL={1} AVG_GROUP_N={2}".format(busy_factor, AVG_PEOPLE_PER_HOUR, self.AVG_GROUP_N))
                if busy_factor == 0:  # se non c'è nessuno a quell'ora, vai alla prossima. non estrarre da var casuale.
                    yield self.env.timeout(60)
                    continue

                # print(hour, busy_factor * AVG_PEOPLE_PER_HOUR / self.AVG_GROUP_N)
                t = random.expovariate(busy_factor * AVG_PEOPLE_PER_HOUR / self.AVG_GROUP_N / 60)
                # logger.debug(t)
                # print(t)
                yield self.env.timeout(t)

                # logger.debug(" * Alle ore %f arriva una persona" % (self.env.now))
                # todo le persone vanno in locali casuali
                place = random.choice(self.places)
                # todo quante persone ci sono in un tavolo?
                num_seats = random.choices(list(self.group_n.keys()), weights=list(self.group_n.values()))[0]

                if place.available_seats:
                    data = {'place': place, 'num_seats': num_seats, 'id': self.placegoer_progressive,
                            'time': self.env.now, 'delta_t': t}

                    self.env.process(person.placegoer(self.env, data))
                    self.placegoer_progressive += 1

                    simlog.add_data(data)
                prevh = hour
        finally:
            logger.info('Simulation ended')
            return simlog

    def start(self):
        # Start process and run
        simlog = SimulationLog()

        self.env.process(self.customer_arrivals(simlog))
        self.env.run(until=SIM_TIME)

        return simlog

    @staticmethod
    def animate(snapshots, name, nSeconds=2):
        import matplotlib.animation as animation

        fps = 60
        # nSeconds = 2
        # snapshots = [np.random.rand(5, 5) for _ in range(nSeconds * fps)]

        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=(8, 8))

        a = snapshots[0]
        im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1, cmap='gist_gray')

        def animate_func(i):
            if i % fps == 0:
                print('.', end='')

            im.set_array(snapshots[i])
            return [im]

        anim = animation.FuncAnimation(
            fig,
            animate_func,
            frames=nSeconds * fps,
            interval=1000 / fps,  # in ms
        )

        anim.save(name + '.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

        print('Done!')

        plt.show()  # Not required, it seems!

    def analyze(self, simlog: SimulationLog):
        # print(simlog.get_df().mean())
        df = simlog.get_df()
        df = df[['num_seats', 'time', 'time_exit', 'went_away']]
        df.hist(bins=5)
        plt.show()

        # for place in self.places:
        #     if place.full:
        #         print('Movie "{0}" sold out {1} minutes after ticket counter '
        #               'opening.'.format(place.name, place.when_full))
        #        print('  Number of people leaving queue when film sold out: %s' %
        #              place.went_away)

        n_people_by_hour = dict()
        n_people_left_by_hour = dict()
        n_people_okay_by_hour = dict()
        for hour in range(int(SIM_TIME / 60)):
            n_people = df[
                (hour * 60 < df['time']) & (df['time'] < (hour + 1) * 60)].sum()['num_seats']
            n_people_left = df[
                (hour * 60 < df['time']) & (df['time'] < (hour + 1) * 60) & (df['went_away'] == 1)].sum()['num_seats']

            n_people_okay = df[
                (hour * 60 < df['time']) & (df['time'] < (hour + 1) * 60) & (df['went_away'] == 0) & (
                            df['time_exit'] is not None)].sum()['num_seats']

            print(
                "Between {0} and {1}, there were {2} customers ({3}/{4})".format(hour, hour + 1, n_people,
                                                                                 n_people_okay, n_people_left))
            n_people_by_hour[hour] = n_people
            n_people_left_by_hour[hour] = n_people_left

            n_people_okay_by_hour[hour] = n_people_okay

        print(n_people_by_hour, n_people_okay_by_hour, n_people_left_by_hour)

        plt.bar(n_people_okay_by_hour.keys(), n_people_okay_by_hour.values())
        plt.bar(n_people_left_by_hour.keys(), n_people_left_by_hour.values(),
                bottom=list(n_people_okay_by_hour.values()))

        plt.plot([0, SIM_TIME / 60], [sim.get_total_seats() * (60 / person.DINING_TIME)] * 2, linestyle='dashed',
                 c='green',
                 markersize=12)

        plt.show()
        # print(df[bool(0 < (df['time'] / 60) < 0 + 1)])
        # print(df[df['time'] < 250])

    def add_place(self, name='Unnamed place', available_seats=50, tables4=5, tables6=3, tables8=3):
        self.places.append(Place(self.env, name, available_seats, tables4=tables4, tables6=tables6, tables8=tables8))


if __name__ == '__main__':
    logger = logging_setup()
    # initialize the simulator
    logger.debug("Initializing simulator...")
    sim = Simulator()
    logger.debug("Adding places...")

    small = [4, 2, 2]
    medium = [8, 3, 3]
    large = [12, 4, 4]

    tables4 = [4, 8, 12]
    tables6 = [2, 3, 4]
    tables8 = [2, 3, 4]

    occurrence = [0.3, 0.6, 0.1]

    for x in range(30):
        sim.add_place(name='Locale %d' % x, tables4=random.choices(tables4, weights=occurrence)[0],
                      tables6=random.choices(tables6, weights=occurrence)[0],
                      tables8=random.choices(tables8, weights=occurrence)[0])

    # start the simulation (in a lazy way?)
    sl = sim.start()

    # analyze the results (computing now?)
    sim.analyze(sl)
