from dataset import PrimusDataset
from pathlib import Path

primus_path = Path(Path.home(), Path("Data/sheet-music/primus"))

if __name__ == "__main__":
  primus_dataset = PrimusDataset(primus_path)
  print("DONE!")