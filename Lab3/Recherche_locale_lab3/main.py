import argparse

import local_search as ls
import magic_square as ms
import local_search_with_restarts as lswr
import simulated_annealing as sa

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scenario', type=str, default="")

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_arguments()

    seed = 0

    if args.scenario == "S1":
        print("Local Search on magic square")
        print("size: 3 x 3")
        print("neigbourhood: 2-swap")
        print("validity function: only the improving neighbours")
        print("evaluation function: number of conflicts")

        magic_square = ms.MagicSquare(3)
        solution = ls.local_search(magic_square, ls.generate_two_swap_neighboorhood, ls.is_improving_validity_function, ls.eval_n_conflict, seed)

    elif args.scenario == "S2":
        print("Local Search on magic square")
        print("size: 3 x 3")
        print("neigbourhood: 2-swap")
        print("validity function: only the improving neighbours")
        print("evaluation function: number of weighted conflicts")

        magic_square = ms.MagicSquare(3)
        solution = ls.local_search(magic_square, ls.generate_two_swap_neighboorhood, ls.is_improving_validity_function, ls.eval_degree_conflict, seed)

    elif args.scenario == "S3":
        print("Local Search on magic square")
        print("size: 5 x 5")
        print("neigbourhood: 2-swap")
        print("validity function: only the improving neighbours")
        print("evaluation function: number of weighted conflicts")

        magic_square = ms.MagicSquare(5)
        solution = ls.local_search(magic_square, ls.generate_two_swap_neighboorhood, ls.is_improving_validity_function, ls.eval_degree_conflict, seed)

    elif args.scenario == "S4":
        print("Local Search on magic square")
        print("size: 5 x 5")
        print("neigbourhood: 3-swap")
        print("validity function: only the improving neighbours")
        print("evaluation function: number of weighted conflicts")


        magic_square = ms.MagicSquare(5)
        solution = ls.local_search(magic_square, ls.generate_three_swap_neighboorhood, ls.is_improving_validity_function, ls.eval_degree_conflict, seed)

    elif args.scenario == "S5":
        print("Local Search with restart on magic square")
        print("size: 5 x 5")
        print("neigbourhood: 3-swap")
        print("validity function: only the improving neighbours")
        print("evaluation function: number of weighted conflicts")
        print("n_restart: 10")


        magic_square = ms.MagicSquare(5)
        n_restart = 10
        solution = lswr.local_search_with_restart(magic_square, ls.generate_two_swap_neighboorhood, ls.is_improving_validity_function, ls.eval_degree_conflict, seed, n_restart)

    elif args.scenario == "S6":
        print("Local Search with restart on magic square")
        print("size: 5 x 5")
        print("neigbourhood: 2-swap")
        print("validity function: only the improving neighbours")
        print("evaluation function: number of weighted conflicts")
        print("n_restart: 100")


        magic_square = ms.MagicSquare(5)
        n_restart = 100
        solution = lswr.local_search_with_restart(magic_square, ls.generate_two_swap_neighboorhood, ls.is_improving_validity_function, ls.eval_degree_conflict, seed, n_restart)

    elif args.scenario == "S7":
        print("Local Search with restart on magic square")
        print("size: 6 x 6")
        print("neigbourhood: 2-swap")
        print("validity function: only the improving neighbours")
        print("evaluation function: number of weighted conflicts")
        print("n_restart: 100")


        magic_square = ms.MagicSquare(6)
        n_restart = 100
        solution = lswr.local_search_with_restart(magic_square, ls.generate_two_swap_neighboorhood, ls.is_improving_validity_function, ls.eval_degree_conflict, seed, n_restart)

    elif args.scenario == "S8":
        print("Simulated annealing on magic square")
        print("size: 6 x 6")
        print("neigbourhood: 2-swap")
        print("t_init =  10, alpha = 0.9")
        print("validity function: accep all neighbours")
        print("evaluation function: number of weighted conflicts")
        print("n_restart: 100")


        magic_square = ms.MagicSquare(6)
        t_init = 10
        alpha = 0.9
        n_iter = 10000
        solution = sa.simulated_annealing(magic_square, ls.generate_two_swap_neighboorhood, ls.accept_all_neighboors, ls.eval_degree_conflict, t_init, alpha, n_iter, seed)

    elif args.scenario == "S9":
        print("Implement your own local search methods and try to solve the 6x6 magic square !")