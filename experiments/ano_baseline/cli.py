"""Shared calendar-period arguments for ANO fitting commands."""

import argparse


def month_group(value):
    """Parse one comma-separated month group; the model validates the partition."""
    try:
        months = [int(month.strip()) for month in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use comma-separated integer months, for example 12,1,2."
        ) from exc
    if not months or any(month < 1 or month > 12 for month in months):
        raise argparse.ArgumentTypeError("Month numbers must be between 1 and 12.")
    return months


def add_mean_period_arguments(parser):
    parser.add_argument(
        "--mean-period", default="all",
        choices=["day", "month", "all", "season", "seasonal"],
    )
    parser.add_argument(
        "--season-months", nargs="+", type=month_group, default=None,
        metavar="MONTHS",
        help="Month groups for season/seasonal; default: 12,1,2 3,4,5 6,7,8 9,10,11.",
    )


def validate_mean_period_arguments(parser, args):
    if args.season_months is not None and args.mean_period not in {"season", "seasonal"}:
        parser.error("--season-months requires --mean-period season.")
    # Validate before opening weather datasets or starting an expensive experiment.
    from lib.models.ano_corr_accumulator import ANOCorrector

    try:
        ANOCorrector(time_mean_period=args.mean_period, season_months=args.season_months)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return args
