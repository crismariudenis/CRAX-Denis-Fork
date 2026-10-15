import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import argparse  # type: ignore
import json  # type: ignore
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize output logs")

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path of the out file to be visualized",
    )

    return parser.parse_args()


def open_and_parse_file(file_path: str):
    """
    This function is to read and put the logs into dictionary format
    Args:
        file_path (string): Path of the log we want to process.
    """
    current_stage = None
    current_step = None

    path_obj = Path(file_path)
    job_id = path_obj.stem.split("_")[-1]
    print(f"Processing Job ID: {job_id}")

    data_dictionary = {}

    with open(file_path, "r") as f:
        content = f.readlines()

        for line in content:
            if line.startswith("[Stage "):
                current_stage = int(line.split()[1].split("/")[0])

                if current_stage not in data_dictionary:
                    data_dictionary[current_stage] = {}

            elif line.startswith("Step "):
                current_step = int(line.split()[1].rstrip(":"))
                if current_step not in data_dictionary[current_stage]:
                    data_dictionary[current_stage][current_step] = {}

            elif (
                ":" in line
                and "/" in line
                and current_stage is not None
                and current_step is not None
            ):
                parts = line.strip().split(":")
                if len(parts) == 2:
                    try:
                        metric_name = parts[0].strip()
                        metric_value = float(parts[1].strip())
                        data_dictionary[current_stage][current_step][metric_name] = (
                            metric_value
                        )
                    except ValueError:
                        pass

        output_dir = Path("Data")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / f"data_{job_id}.json", "w") as f:
            json.dump(data_dictionary, f, indent=4)

        return data_dictionary, job_id


def draw_from_dict(data_dict, output_dir="Plots", job_id=""):
    """
    Plots every metric separated by curriculum stage and saves each plot as an image.

    Args:
        data_dict (dict): Nested dictionary {stage: {step: {metric: value}}}
        output_dir (str or Path): Target directory to save the plots.
        job_id (str): Optional identifier for naming folders and titles.
    """
    if not data_dict:
        print("Data dictionary is empty.")
        return

    # Subfolder for this job's plots
    save_dir = Path(output_dir)
    if job_id:
        save_dir = save_dir / f"job_{job_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert stage keys to sorted integers
    stages = sorted([int(k) for k in data_dict.keys()])

    # Find all unique metrics across all stages and steps
    all_metrics = set()
    for stage_data in data_dict.values():
        for step_metrics in stage_data.values():
            all_metrics.update(step_metrics.keys())

    all_metrics = sorted(list(all_metrics))
    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for metric in all_metrics:
        fig, axes = plt.subplots(
            1, len(stages), figsize=(5.5 * len(stages), 4.5), sharey=False
        )
        if len(stages) == 1:
            axes = [axes]

        for idx, stage in enumerate(stages):
            ax = axes[idx]
            stage_key = stage if stage in data_dict else str(stage)
            stage_data = data_dict.get(stage_key, {})

            # Sort steps numerically
            steps = sorted([int(st) for st in stage_data.keys()])

            x_steps = []
            y_vals = []
            for st in steps:
                st_key = st if st in stage_data else str(st)
                if metric in stage_data[st_key]:
                    x_steps.append(st)
                    y_vals.append(stage_data[st_key][metric])

            color = stage_colors[idx % len(stage_colors)]

            if x_steps:
                # Format steps in Millions for better readability if large
                max_step = max(x_steps)
                use_millions = max_step >= 1_000_000
                x_plot = [s / 1e6 for s in x_steps] if use_millions else x_steps

                # If data points are sparse (e.g. eval metrics), show markers
                show_markers = len(x_steps) <= 25

                ax.plot(
                    x_plot,
                    y_vals,
                    color=color,
                    marker="o" if show_markers else None,
                    markersize=4,
                    linewidth=1.8,
                    label=f"Stage {stage}",
                )
                ax.set_title(f"Stage {stage}", fontsize=12, fontweight="bold")
                ax.set_xlabel(
                    "Steps (Millions)" if use_millions else "Steps", fontsize=10
                )
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(loc="best")
            else:
                ax.set_title(f"Stage {stage} (No Data)", fontsize=12)
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
                ax.grid(True, linestyle="--", alpha=0.3)

            if idx == 0:
                ax.set_ylabel(metric, fontsize=10, fontweight="bold")

        title_suffix = f" (Job {job_id})" if job_id else ""
        fig.suptitle(f"{metric}{title_suffix}", fontsize=13, fontweight="bold")
        fig.tight_layout()

        # Sanitize filename by replacing '/' with '_'
        safe_filename = metric.replace("/", "_") + ".png"
        fig.savefig(save_dir / safe_filename, dpi=150)
        plt.close(fig)

    print(f"{len(all_metrics)} plots saved: {save_dir}")


def draw_from_json(json_file_path, output_dir="Plots"):
    """
    Reads a saved JSON file and plots all metrics.

    Args:
        json_file_path (str or Path): Path to the JSON data file.
        output_dir (str or Path): Directory to save plots.
    """
    path_obj = Path(json_file_path)
    with open(path_obj, "r") as f:
        data_dict = json.load(f)

    # Try extracting job_id from filename, e.g., data_27746927.json
    job_id = ""
    parts = path_obj.stem.split("_")
    if len(parts) >= 2:
        job_id = parts[-1]

    draw_from_dict(data_dict, output_dir=output_dir, job_id=job_id)


def visualize_out_file(file_path: str, output_dir="Plots"):
    """
    Parses a .out file and directly plots all metrics.
    """
    data_dict, job_id = open_and_parse_file(file_path)
    draw_from_dict(data_dict, output_dir=output_dir, job_id=job_id)


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.path)
    if input_path.suffix == ".json":
        draw_from_json(input_path)
    else:
        visualize_out_file(args.path)
