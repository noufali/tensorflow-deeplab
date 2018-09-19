import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser for tensorflow-deeplab")
        subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")

        # demo
        demo_arg = subparsers.add_parser("demo", help="parser for evaluation/stylizing arguments")
        demo_arg.add_argument("--image", type=str, default="images/",
                                help="path to image")

    def parse(self):
        return self.parser.parse_args()
