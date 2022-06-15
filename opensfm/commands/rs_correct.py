from opensfm.actions import rs_correct

from . import command
import argparse
from opensfm.dataset import DataSet


class Command(command.CommandBase):
    name = "rs_correct"
    help = "Apply rolling shutter correction to a reconstruction"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        rs_correct.run_dataset(dataset, args.output, args.output_tracks, args.rolling_shutter_readout)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output", help="file name where to store the corrected reconstruction"
        )
        parser.add_argument(
            "--output-tracks", help="file name where to store corrected tracks"
        )
        parser.add_argument(
            "--rolling-shutter-readout", help="Rolling shutter sensor readout time (ms)", type=float, default=30.0
        )
