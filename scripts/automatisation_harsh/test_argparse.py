import argparse


def main():

    parser = argparse.ArgumentParser()

    # Optional arguments:
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )  # pour éviter de devoir écrire un int après --verbose : on met une action "store_true "
    parser.add_argument("-s", "--starting_node", help="start node", type=int, default=0)
    parser.add_argument("-t", "--target_node", help="end node", type=int, default=1)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--filename", type=str, help="filename for adj matrix")
    group.add_argument(
        "-n",
        "--number_nodes_edges",
        help="number of nodes and edges",
        type=int,
        nargs="+",
        default=[4, 4],
    )
    parser.add_argument("-r", "--reps", type=int, required=True, help="")
    # Add a list:
    parser.add_argument(
        "-a", "--alphas", nargs="+", type=int, help="List of alphas", default=[1.1]
    )

    # parse_args method : returns data from options
    args = parser.parse_args()

    print(args.alphas)
    print(type(args.alphas))
    if args.verbose:
        print("verbose")
    else:
        print("nothing")


if __name__ == "__main__":
    main()
