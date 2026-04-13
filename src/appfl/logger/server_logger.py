import os
import csv
import pathlib
import logging
from typing import List, Dict, Union

class ServerAgentFileLogger:
    def __init__(self, file_dir: str="", file_name: str="") -> None:
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s server]: %(message)s')
        file_name += "_server"
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.csv_file = None
        self.csv_writer = None
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        self.logger.addHandler(s_handler)
        if file_dir != "" and file_name != "":
            if not os.path.exists(file_dir):
                pathlib.Path(file_dir).mkdir(parents=True)
            uniq = 1
            real_file_name = f"{file_dir}/{file_name}.txt"
            while os.path.exists(real_file_name):
                real_file_name = f"{file_dir}/{file_name}_{uniq}.txt"
                uniq += 1
            self.logger.info(f"Logging to {real_file_name}")
            f_handler = logging.FileHandler(real_file_name)
            f_handler.setLevel(logging.INFO)
            f_handler.setFormatter(fmt)
            self.logger.addHandler(f_handler)
            csv_file_name = real_file_name[:-4] + ".csv"
            self.csv_file = open(csv_file_name, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.logger.info(f"Structured metrics will be saved to {csv_file_name}")

    def info(self, info: str) -> None:
        self.logger.info(info)

    def log_title(self, titles: List) -> None:
        self.titles = titles
        title = " ".join(["%14s" % t for t in titles])
        self.logger.info(title)
        if self.csv_writer is not None:
            self.csv_writer.writerow(titles)
            self.csv_file.flush()

    def log_content(self, contents: Union[Dict, List]) -> None:
        if not isinstance(contents, dict) and not isinstance(contents, list):
            raise ValueError("Contents must be a dictionary or list")
        if not isinstance(contents, list):
            for key in contents.keys():
                if key not in self.titles:
                    raise ValueError(f"Title {key} is not defined")
            contents = [contents.get(key, "") for key in self.titles]
        else:
            if len(contents) != len(self.titles):
                raise ValueError("Contents and titles must have the same length")
        length = [max(len(str(t)), 14) for t in self.titles]
        content = " ".join(["%*s" % (l, c) if not isinstance(c, float) else "%*.4f" % (l, c) for l, c in zip(length, contents)])
        self.logger.info(content)
        if self.csv_writer is not None:
            self.csv_writer.writerow(contents)
            self.csv_file.flush()

    def __del__(self) -> None:
        if hasattr(self, "csv_file") and self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.close()
