import random

import matplotlib.pyplot as plt
import numpy as np
import simpy
import person
from place import Place
# from logging import info, warning, error, debug, critical
import logging
import pandas as pd
from typing import List

# create logger with 'spam_application'
from simulationlog import SimulationLog

RANDOM_SEED = None
# RANDOM_SEED = 42
SIM_TIME = (24 + 3) * 60
AVG_PEOPLE_PER_HOUR = 2500
LONG_TAIL_FACTOR = 3


def logging_setup():
    logger = logging.getLogger('affluence_simulator')
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('tuttecose.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class Simulator:
    def __init__(self, app_usage=0.05):
        self.places = []
        self.popular_places = []
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

        self.group_n = {1: 0, 2: 6, 3: 0, 4: 44, 5: 0, 6: 34, 7: 0, 8: 16}
        self.AVG_GROUP_N = sum([(n * p) / 100 for n, p in zip(self.group_n.keys(), self.group_n.values())])
        self.APP_USAGE = app_usage

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

                t = random.expovariate(busy_factor * AVG_PEOPLE_PER_HOUR / self.AVG_GROUP_N / 60)
                # logger.debug(t)
                # print(t)
                yield self.env.timeout(t)

                # le persone vanno in locali casuali (power law)
                place = random.choices(population=self.places, weights=self.popular_places)[0]
                # place = random.choice(self.places)
                # quante persone ci sono in un tavolo?
                num_seats = random.choices(list(self.group_n.keys()), weights=list(self.group_n.values()))[0]

                # ha l'app?
                has_app = random.choices([True, False], weights=[self.APP_USAGE, 1 - self.APP_USAGE])[0]

                # if place.available_seats:
                data = {'place': place, 'num_seats': num_seats, 'id': self.placegoer_progressive,
                        'time': self.env.now, 'delta_t': t, 'has_app': has_app}

                self.env.process(person.placegoer(self.env, data, self.places))
                self.placegoer_progressive += 1

                simlog.add_data(data)
                prevh = hour
        except Exception as e:
            print(e)
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
        df = simlog.get_df()
        df = df[['num_seats', 'time', 'time_exit', 'went_away', 'waiting_time']]
        # df.hist(bins=5)
        # plt.show()

        hours_to_show = [23, 24, 25]

        n_people_by_hour = dict()
        n_people_left_by_hour = dict()
        n_people_okay_by_hour = dict()
        all_waiting_time_histogram_data_okay = []
        all_waiting_time_histogram_data_left = []

        for hour in range(int(SIM_TIME / 60)):
            df_hour = df[
                (hour * 60 < df['time']) & (df['time'] < (hour + 1) * 60)]

            n_people = df_hour.sum()['num_seats']

            waiting_time_histogram_data_okay = []
            waiting_time_histogram_data_left = []

            # avg_waiting_time = (df_hour['waiting_time'] * df_hour['num_seats'] / n_people)
            n_people_left = df_hour[df_hour['went_away'] == 1].sum()['num_seats']

            n_people_okay = df_hour[(df_hour['went_away'] == 0) & (
                    df_hour['time_exit'] is not None)].sum()['num_seats']

            # print(
            #    "Between {0} and {1}, there were {2} customers ({3}/{4})".format(hour, hour + 1, n_people,
            # n_people_okay, n_people_left))
            n_people_by_hour[hour] = n_people
            n_people_left_by_hour[hour] = n_people_left
            n_people_okay_by_hour[hour] = n_people_okay

            if True:  # hour in hours_to_show:

                if not df_hour.empty:
                    for data_point in df_hour.iterrows():
                        n = int(data_point[1]['num_seats'])
                        if np.isnan(data_point[1]['waiting_time']):
                            continue
                        if data_point[1]['went_away'] == 1:
                            waiting_time_histogram_data_left.extend([data_point[1]['waiting_time']] * n)
                        else:
                            waiting_time_histogram_data_okay.extend([data_point[1]['waiting_time']] * n)

                    all_waiting_time_histogram_data_left.extend(waiting_time_histogram_data_left)
                    all_waiting_time_histogram_data_okay.extend(waiting_time_histogram_data_okay)

                # plt.hist([waiting_time_histogram_data_okay, waiting_time_histogram_data_left], bins=20, histtype='bar',
                #          stacked=True)

                # plt.title(
                #     "Tempo attesa, ore {0}. \nN_serviti = {4} Serviti={1:.2%}\n Mediana attesa={2:.2f} \nMediana attesa rimasti={3:.2f}".format(
                #         hour,
                #         n_people_okay / (n_people_okay + n_people_left),
                #         np.median(np.array(waiting_time_histogram_data_okay + waiting_time_histogram_data_left)),
                #         np.median(waiting_time_histogram_data_okay),
                #         n_people_okay
                #     )
                # )
                # plt.show()

        # print(n_people_by_hour, n_people_okay_by_hour, n_people_left_by_hour)

        # plt.bar(n_people_okay_by_hour.keys(), n_people_okay_by_hour.values())
        # plt.bar(n_people_left_by_hour.keys(), n_people_left_by_hour.values(),
        #         bottom=list(n_people_okay_by_hour.values()))
        # plt.title("Clienti per fascia oraria")

        # plt.plot([0, SIM_TIME / 60], [sim.get_total_seats() * (60 / 60)] * 2, linestyle='dashed',
        #          c='green',
        #         markersize=12)

        # plt.show()
        # print(df[bool(0 < (df['time'] / 60) < 0 + 1)])
        # print(df[df['time'] < 250])

        n_people_total = df.sum()['num_seats']
        n_people_okay_total = df[(df['went_away'] != 1)].sum()['num_seats']
        n_people_left_total = df[df['went_away'] == 1].sum()['num_seats']

        return SingleSimulationResult(n_people_by_hour=n_people_by_hour, n_people_okay_by_hour=n_people_okay_by_hour,
                                      n_people_left_by_hour=n_people_left_by_hour, n_people=n_people_total,
                                      n_people_okay=n_people_okay_total, n_people_left=n_people_left_total,
                                      waiting_times_okay=all_waiting_time_histogram_data_okay,
                                      waiting_times_left=all_waiting_time_histogram_data_left)

    def add_place(self, name='Unnamed place', available_seats=50, tables4=5, tables6=3, tables8=3):
        self.places.append(Place(self.env, name, available_seats, tables4=tables4, tables6=tables6, tables8=tables8))

    def generate_popular_places(self):
        # self.popular_places = 1 - np.random.power(LONG_TAIL_FACTOR, len(self.places)) + 0.1
        self.popular_places = 1 - np.random.power(LONG_TAIL_FACTOR, len(self.places)) + 0.05
        # print(max(self.popular_places)/min(self.popular_places))


class SingleSimulationResult:
    def __init__(self, n_people_by_hour, n_people_okay_by_hour, n_people_left_by_hour, n_people, n_people_okay,
                 n_people_left, waiting_times_okay, waiting_times_left):
        self.n_people_by_hour = n_people_by_hour
        self.n_people_okay_by_hour = n_people_okay_by_hour
        self.n_people_left_by_hour = n_people_left_by_hour
        self.n_people = n_people
        self.n_people_okay = n_people_okay
        self.n_people_left = n_people_left
        self.waiting_times_okay = waiting_times_okay
        self.waiting_times_left = waiting_times_left


def monte_carlo_simulation(app_usage=0.05, N=100):
    simulations = []
    for i in range(N):
        if i % (N / 10) == 0:
            logger.info("Simulations progress: {0:.1%}".format(i / N))
        sim = Simulator(app_usage=app_usage)
        logger.debug("Adding places...")

        tables4 = [4, 8, 12]
        tables6 = [2, 4, 6]
        tables8 = [2, 3, 4]

        occurrence = [0.3, 0.6, 0.1]

        for x in range(30):
            sim.add_place(name='Locale %d' % x, tables4=random.choices(tables4, weights=occurrence)[0],
                          tables6=random.choices(tables6, weights=occurrence)[0],
                          tables8=random.choices(tables8, weights=occurrence)[0])
        sim.generate_popular_places()
        # start the simulation
        sl = sim.start()

        # analyze the results
        res = sim.analyze(sl)

        simulations.append(res)

    return simulations


def analyze_simulations(sims: List[SingleSimulationResult], desc):
    N = len(sims)
    total_okay_customers = []
    total_left_customers = []
    total_customers = []
    waiting_times_okay = []
    waiting_times_left = []
    waiting_times_all = []

    for sim in sims:
        total_okay_customers.append(sim.n_people_okay)
        total_left_customers.append(sim.n_people_left)
        total_customers.append(sim.n_people)
        waiting_times_okay.extend(sim.waiting_times_okay)
        waiting_times_left.extend(sim.waiting_times_left)
        waiting_times_all.extend(sim.waiting_times_okay)
        waiting_times_all.extend(sim.waiting_times_left)

    print(desc)
    print("Total customers: avg={0}, std={1}".format(np.mean(total_customers), np.std(total_customers)))
    print("Customers served: avg={0}, std={1} (%={2:.2%})".format(
        np.mean(total_okay_customers), np.std(total_okay_customers),
        np.mean(np.array(total_okay_customers) / np.array(total_customers))))

    print("Customers lost: avg={0}, std={1} (%={2:.2%})".format(
        np.mean(total_left_customers), np.std(total_left_customers),
        np.mean(np.array(total_left_customers) / np.array(total_customers))))

    print("Avg waiting time ALL: avg={0}, std={1}".format(np.mean(waiting_times_all), np.std(waiting_times_all)))

    result = dict()
    result['total_okay_customers_avg'] = np.mean(total_okay_customers)
    result['total_okay_customers_std'] = np.std(total_okay_customers)
    result['total_left_customers_avg'] = np.mean(total_left_customers)
    result['total_left_customers_std'] = np.std(total_left_customers)
    result['total_customers_avg'] = np.mean(total_customers)
    result['total_customers_std'] = np.std(total_customers)
    result['total_waiting_time_okay_avg'] = np.mean(waiting_times_okay)
    result['total_waiting_time_okay_std'] = np.std(waiting_times_okay)
    result['total_waiting_time_left_avg'] = np.mean(waiting_times_left)
    result['total_waiting_time_left_std'] = np.std(waiting_times_left)
    result['total_waiting_time_all_avg'] = np.mean(waiting_times_all)
    result['total_waiting_time_all_std'] = np.std(waiting_times_all)

    return result


def plot_analysis(x, avg, std, desc):
    customer_served_all_avg = np.array([d[avg] for d in multi_sims])
    customer_served_all_std = np.array([d[std] for d in multi_sims])

    # customer_served_all_avg = customer_served_all_avg.reshape(-1)
    # customer_served_all_std = customer_served_all_std.reshape(-1)

    f, ax = plt.subplots(1)

    ax.errorbar(x, customer_served_all_avg, yerr=customer_served_all_std)
    ax.set_ylim(ymin=0)
    plt.title(desc)
    plt.show()


if __name__ == '__main__':
    logger = logging_setup()
    # initialize the simulator
    logger.info("Initializing...")

    N = 50
    resolution = 5


    multi_sims = []
    x = np.geomspace(0.1, 1.1, resolution) - 0.1
    for app_usage in x:
        logger.info("Running {0} simulations for scenario app_usage={1}...".format(N, app_usage))

        simulations = monte_carlo_simulation(app_usage=app_usage, N=N)

        res_simulations = analyze_simulations(simulations, "Uso app {0:%}%".format(app_usage))
        multi_sims.append(res_simulations)

    plot_analysis(x, 'total_okay_customers_avg', 'total_okay_customers_std', "Clienti serviti")
    plot_analysis(x, 'total_left_customers_avg', 'total_left_customers_std', "Clienti andati via")
    plot_analysis(x, 'total_waiting_time_okay_avg', 'total_waiting_time_okay_std',
                  "Tempo di attesa medio per clienti serviti")
    plot_analysis(x, 'total_waiting_time_left_avg', 'total_waiting_time_left_std',
                  "Tempo di attesa medio per clienti andati via")
    plot_analysis(x, 'total_waiting_time_all_avg', 'total_waiting_time_all_std',
                  "Tempo di attesa medio per clienti")
