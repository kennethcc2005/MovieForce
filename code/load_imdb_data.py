from collections import defaultdict


def load_imdb_data(filename):
    '''
    filename: name of imdb edge data file

    Read in the data and create two dictionaries of adjacency lists, one for
    the actors and one for the movies.
    '''
    f = open(filename)
    actors = defaultdict(set)
    movies = defaultdict(set)
    for line in f:
        actor, movie = line.strip().split('\t')
        actors[actor].add(movie)
        movies[movie].add(actor)
    f.close()
    return actors, movies
