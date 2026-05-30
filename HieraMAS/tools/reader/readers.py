#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Minimal readers used by the camera-ready experiment scripts."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from HieraMAS.utils.log import logger


class Reader(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """Parse a file into text."""


class JSONReader(Reader):
    @staticmethod
    def parse_file(file_path: Path) -> list:
        logger.info(f"Reading JSON file from {file_path}.")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def parse(self, file_path: Path) -> str:
        return str(self.parse_file(file_path))


class JSONLReader(Reader):
    @staticmethod
    def parse_file(file_path: Path) -> list:
        logger.info(f"Reading JSON Lines file from {file_path}.")
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def parse(file_path: Path) -> str:
        lines = JSONLReader.parse_file(file_path)
        return "\n".join(str(line) for line in lines)


READER_MAP = {
    ".json": JSONReader(),
    ".jsonl": JSONLReader(),
}


class FileReader:
    def set_reader(self, suffix: str) -> None:
        if suffix not in READER_MAP:
            raise ValueError(f"Unsupported file suffix in minimal reader: {suffix}")
        self.reader = READER_MAP[suffix]
        logger.info(f"Setting Reader to {type(self.reader).__name__}")

    def read_file(self, file_path: Path) -> str:
        suffix = Path(file_path).suffix
        self.set_reader(suffix)
        return self.reader.parse(Path(file_path))


class GeneralReader:
    def __init__(self):
        self.file_reader = FileReader()
        self.name = "Minimal File Reader"
        self.description = "Reads JSON and JSONL files for experiment datasets."

    def read(self, task, file):
        return self.file_reader.read_file(Path(file))
