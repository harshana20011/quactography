import argparse

from quactography.visu.dist_prob import plot_distribution_of_probabilities


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "input_files",
        nargs="+",
        help="List of input files to plot distribution",
    )
    p.add_argument(
        "visual_dist_output_file_total",
        help="Output file name for visualisation",
        type=str,
    )
    p.add_argument(
        "visual_dist_output_file_selected",
        help="Output file name for visualisation",
        type=str,
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    for i in range(len(args.input_files)):
        plot_distribution_of_probabilities(
            args.input_files[i],
            args.visual_dist_output_file_total + "_" + str(i),
            args.visual_dist_output_file_selected + "_" + str(i),
        )


if __name__ == "__main__":
    main()