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

MAX_PATIENCE = 30
MIN_PATIENCE = 5

DINING_TIME = 30

def placegoer(env, data):
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

    logger = logging.getLogger('affluence_simulator')
    logger.debug(
        "(%.2f) [%d] wants %d seats (%d left) at [%s] " % (env.now, id, num_seats, place.available_seats, place.name))

    with place.reception.request() as my_turn:
        # logger.debug("[id=%d] (%d) è il mio turno per prendere %d biglietti al locale %s!" % (id, env.now, num_seats, place.name))
        # Wait until its our turn or until the movie is sold out
        # result = yield my_turn | place.full

        patience = (1 - np.random.power(a=5)) * MAX_PATIENCE + MIN_PATIENCE
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

        # Check if enough tickets left.
        # if place.available_seats < num_seats:
        # Moviegoer leaves after some discussion
        #    logger.debug("[id=%d] >>:(" % id)
        #    # todo: mettersi in coda per il tavolo, che è una risorsa con capacità variabile.

        #    yield env.timeout(0.5)
        #    return

        # serve customer
        yield env.timeout(1)

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
        yield env.timeout(DINING_TIME)
        # place.available_seats += num_seats
        logger.debug(
            "(%.2f) [%d] freed %d seats (%d left) at [%s] " % (
            env.now, id, num_seats, place.available_seats, place.name))
        data['time_exit'] = env.now
