"""Create an API key: prints the raw key ONCE and the record for API_KEYS_JSON (spec §2.2 #5).

Usage:
    python scripts/make_api_key.py --id demo --rpm 6 --daily-budget 0.10
"""

import argparse
import json
import secrets

from atlas.security.auth import hash_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="key identifier (e.g. demo)")
    parser.add_argument("--rpm", type=int, default=60)
    parser.add_argument("--daily-budget", type=float, default=0.25)
    parser.add_argument("--scopes", default="ask")
    args = parser.parse_args()

    raw = f"atlas_{secrets.token_urlsafe(32)}"
    record = {
        "id": args.id,
        "sha256_hash": hash_key(raw, pepper_from_env()),
        "rpm": args.rpm,
        "daily_budget_usd": args.daily_budget,
        "scopes": args.scopes.split(","),
    }
    print("RAW KEY (shown once — store it now):")
    print(raw)
    print("\nAPI_KEYS_JSON entry:")
    print(json.dumps([record]))


def pepper_from_env() -> str:
    import os

    pepper = os.environ.get("API_KEY_PEPPER", "")
    if pepper:
        return pepper
    from atlas.config import get_settings

    return get_settings().api_key_pepper


if __name__ == "__main__":
    main()
