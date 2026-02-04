import argparse
from pathlib import Path
import shutil
import kagglehub


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Download the GTSRB dataset via KaggleHub."
	)
	parser.add_argument(
		"--output-dir",
		"-o",
		required=True,
		help="Target directory that will hold the downloaded dataset.",
	)
	args = parser.parse_args()

	output_dir = Path(args.output_dir).expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Downloading dataset to cache...")
	cache_path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
	
	print(f"Moving files from {cache_path} to {output_dir}...")
	
	shutil.copytree(cache_path, output_dir, dirs_exist_ok=True)

	print("Path to dataset files:", output_dir)


if __name__ == "__main__":
	main()