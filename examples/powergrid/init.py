#!/usr/bin/env python
"""
Power Grid Frequency Control CPS Testbed - init.py

Initializes the SQLite database table used for physical process state.
"""

from minicps.states import SQLiteState
from utils import PATH, SCHEMA, SCHEMA_INIT
from sqlite3 import OperationalError
import os

if __name__ == "__main__":
    if os.path.exists(PATH):
        try:
            os.remove(PATH)
            print("Removed existing {}.".format(PATH))
        except Exception as e:
            print("Warning removing {}: {}".format(PATH, e))

    try:
        SQLiteState._create(PATH, SCHEMA)
        SQLiteState._init(PATH, SCHEMA_INIT)
        print("{} successfully created and initialized.".format(PATH))
    except OperationalError as err:
        print("OperationalError creating {}: {}".format(PATH, err))
