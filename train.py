from lmms.config import parse_args
from lmms.trainer import train_main


if __name__ == "__main__":
    cfg = parse_args()
    train_main(cfg)
