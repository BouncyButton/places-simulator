import collections
import random
import simpy
import typing

from logging import info, warning, error, debug, critical
from simulator import Place
import logging
import numpy as np

RANDOM_SEED = 42
TICKETS = 50  # Number of tickets per movie
SIM_TIME = 120  # Simulate until

# MAX_PATIENCE = 30
PATIENCE = 10
TRAVEL_TIME = 0


# SIM_TIME DINING_TIME = 60

def find_new_place(places, prevbest):
    # min_affluence = min([x.get_affluence_status() for x in places])

    try:
        return random.choice([x for x in places if x.get_affluence_status() <= prevbest - 2])
    except IndexError:
        try:
            return random.choice([x for x in places if x.get_affluence_status() <= prevbest - 1])
        except IndexError:
            return random.choice([x for x in places if x.get_affluence_status() <= prevbest])

def placegoer(env, data, places):
    """A placegoer tries to buy a number of seats (*num_seats*) for
    a certain *bar* in a *city*.

    If the bar becomes sold out, she leaves the city TODO: goes to another bar or waits until it gets free

    If she gets to the counter, she tries to get a number of seats. If not enough
    seats are left, she argues with the teller and leaves.

    If at most one seat is left after the bargoer bought her
    tickets, the *sold out* event for this movie is triggered causing
    all remaining moviegoers to leave.

    TODO people go out of the bar, making it free again.
4
    """
    place = data['place']
    id = data['id']
    num_seats = data['num_seats']
    has_app = data['has_app']

    logger = logging.getLogger('affluence_simulator')
    logger.debug(
        "(%.2f) [%d] wants %d seats (%d left) at [%s] " % (env.now, id, num_seats, place.available_seats, place.name))

    if has_app:
        if place.get_affluence_status() >= 3:
        # if place.available_seats < 10:
            # sceglie un altro posto
            place = find_new_place(places, place.get_affluence_status())
            data['place'] = place
            # travel_time = (np.random.lognormal(sigma=0.33, mean=0)) * TRAVEL_TIME
            # env.timeout(travel_time)
            logger.debug(
                "(%.2f) [%d] chooses to go at [%s] using the app (free=%d)" % (
                env.now, id, place.name, place.available_seats))

    with place.reception.request() as my_turn:
        # logger.debug("[id=%d] (%d) è il mio turno per prendere %d biglietti al locale %s!" % (id, env.now, num_seats, place.name))
        # Wait until its our turn or until the movie is sold out
        # result = yield my_turn | place.full

        patience = (np.random.lognormal(sigma=0.33, mean=0)) * PATIENCE
        data['start_queuing'] = env.now
        # Wait for the counter or abort if out of patience
        result = yield my_turn | env.timeout(patience)

        # Check if it's our turn or if movie is sold out
        if my_turn not in result:
            logger.debug("[id=%d] >:( waited PATIENCE=%d and left (no waiter)" % (id, patience))
            data['went_away'] = 1
            data['why_went_away'] = 'reception_too_long'
            place.went_away += 1
            return
        else:
            data['went_away'] = 0

        # serve customer
        # yield env.timeout(1)

    if num_seats <= 4:
        table_to_pick = place.tables4
    elif 4 < num_seats <= 6:
        table_to_pick = place.tables6
    else:
        table_to_pick = place.tables8

    with table_to_pick.request() as my_table:
        patience_left = max(patience - (env.now - data['start_queuing']), 0)

        # Wait for the counter or abort if out of patience
        result = yield my_table | env.timeout(patience_left)
        # save waiting time (anyway)
        data['waiting_time'] = env.now - data['start_queuing']

        if my_table not in result:
            logger.debug("[id=%d] >:( waited PATIENCE=%d and left (no table)" % (id, patience))
            data['why_went_away'] = 'wait_for_table_too_long'
            data['went_away'] = 1
            return

        # Buy tickets
        # place.available_seats -= num_seats
        logger.debug("(%.2f) [%d] gets %d seats (%d left) at [%s] " % (
            env.now, id, num_seats, place.available_seats, place.name))

        # todo il tempo passato in un locale cambia (sia a seconda del locale, che a seconda delle persone/gruppo)
        yield env.timeout(place.get_random_dining_time())
        # place.available_seats += num_seats
        logger.debug(
            "(%.2f) [%d] freed %d seats (%d left) at [%s] " % (
                env.now, id, num_seats, place.available_seats, place.name))
        data['time_exit'] = env.now
