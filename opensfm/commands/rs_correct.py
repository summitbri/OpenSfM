from opensfm.actions import rs_correct

from . import command
import argparse
from opensfm.dataset import DataSet


class Command(command.CommandBase):
    name = "rs_correct"
    help = "Apply rolling shutter correction to a reconstruction"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        rs_correct.run_dataset(dataset, args.output)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output", help="file name where to store the corrected reconstruction"
        )
