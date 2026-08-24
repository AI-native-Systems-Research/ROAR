#!/usr/bin/env python3
"""Drop all ADRS database tables for a fresh start."""

import os
import psycopg

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/adrs"
)

TABLES = [
    "artifacts",
    "measurements",
    "candidate_edges",
    "candidates",
    "campaigns",
    "systems",
]


def main():
    print(f"Connecting to: {DATABASE_URL}")
    print("WARNING: This will drop all tables and data!")

    response = input("Type 'yes' to confirm: ")
    if response.lower() != "yes":
        print("Aborted.")
        return

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for table in TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"  Dropped: {table}")
        conn.commit()

    print("All tables dropped successfully.")


if __name__ == "__main__":
    main()
