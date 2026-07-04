"""Re-run only the resolve stage of ingest_pbf.

The staging tables tmp_nodes / tmp_edges still hold all 4 PBFs from the
prior streaming pass that crashed during _resolve_into_final.  Replay
the resolve logic without re-streaming.
"""
import psycopg

import config
from .ingest_pbf import _resolve_into_final


def main() -> None:
    with psycopg.connect(config.PG_DSN) as conn:
        with conn.cursor() as cur:
            _resolve_into_final(cur)
        conn.commit()
    print("[resume] resolve complete")


if __name__ == "__main__":
    main()
