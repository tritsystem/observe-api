"""
observe -- command-line client for the hosted OBSERVE Search API.

Real gap this closes: MCP/LangChain/CrewAI wrappers already existed for
agent frameworks that speak those specific protocols, but nothing worked
for a plain shell -- a CI script, a Bash-based agent harness, a human at
a terminal. This is the same shared core.py/commerce.py client, so it's
guaranteed to behave identically to the MCP/tool wrappers, not a
reimplementation with its own bugs.

Usage:
    observe search "where does this handle retrying a failed upload"
    observe search "retryUpload"              # refused by the cost guard
    observe search "retryUpload" --force      # explicit override
    observe repos
    observe balance
    observe commerce-search "waterproof boots for a muddy trail"
    observe commerce-register-seller "Trailhead" "https://.../checkout_sessions"
    observe commerce-add-listings 1 '[{"item_id":"sku-1","name":"Boots","description":"..."}]'
    observe commerce-feedback 1 sku-1 purchased --match-id <from a search result>
    observe commerce-seller-feedback <match_id> fulfilled --rating 5
    observe commerce-reputation
    observe commerce-verify-match <match_id>
    observe commerce-network-stats
"""
import argparse
import json
import os
import sys

from . import commerce, core

API_BASE = os.environ.get("OBSERVE_API_BASE", "https://api.observe-search.online")


def _cmd_search(args):
    # The cost guard lives in core.search() itself, not duplicated here --
    # see core.py's module docstring.
    print(core.search(args.query, k=args.k, repo=args.repo, force=args.force))
    return 0


def _cmd_repos(args):
    import httpx
    try:
        resp = httpx.get(f"{API_BASE}/v1/repos", timeout=10)
    except httpx.RequestError as e:
        print(f"Error: could not reach {API_BASE}: {e}")
        return 1
    print(", ".join(resp.json()["repos"]))
    return 0


def _cmd_balance(args):
    import httpx
    key = os.environ.get("OBSERVE_API_KEY")
    if not key:
        print("Error: OBSERVE_API_KEY is not set.")
        return 1
    try:
        resp = httpx.get(f"{API_BASE}/v1/balance", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    except httpx.RequestError as e:
        print(f"Error: could not reach {API_BASE}: {e}")
        return 1
    if resp.status_code != 200:
        print(f"Error: API returned {resp.status_code}: {resp.text}")
        return 1
    print(f"{resp.json()['credits']} credits remaining.")
    return 0


def _cmd_commerce_search(args):
    print(commerce.commerce_search(args.intent, max_price=args.max_price, category=args.category, k=args.k))
    return 0


def _cmd_commerce_register_seller(args):
    print(commerce.register_seller(args.name, args.checkout_session_url))
    return 0


def _cmd_commerce_add_listings(args):
    try:
        listings = json.loads(args.listings_json)
    except json.JSONDecodeError as e:
        print(f"Error: listings_json is not valid JSON: {e}")
        return 1
    if not isinstance(listings, list):
        print("Error: listings_json must be a JSON array of listing objects.")
        return 1
    print(commerce.add_listings(args.seller_id, listings))
    return 0


def _cmd_commerce_feedback(args):
    print(commerce.report_purchase_feedback(args.seller_id, args.item_id, args.outcome, match_id=args.match_id))
    return 0


def _cmd_commerce_seller_feedback(args):
    print(commerce.report_seller_feedback(args.match_id, args.outcome, rating=args.rating))
    return 0


def _cmd_commerce_reputation(args):
    print(commerce.get_my_reputation())
    return 0


def _cmd_commerce_verify_match(args):
    print(commerce.verify_match(args.match_id))
    return 0


def _cmd_commerce_network_stats(args):
    print(commerce.get_network_stats())
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="observe", description="CLI for the hosted OBSERVE Search API.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("search", help="Semantic code search over the curated corpus.")
    s.add_argument("query")
    s.add_argument("--repo", default=None)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--force", action="store_true", help="Bypass the exact-identifier cost guard.")
    s.set_defaults(func=_cmd_search)

    r = sub.add_parser("repos", help="List currently indexed repos.")
    r.set_defaults(func=_cmd_repos)

    b = sub.add_parser("balance", help="Check remaining API credits.")
    b.set_defaults(func=_cmd_balance)

    cs = sub.add_parser("commerce-search", help="ACP-compatible buyer/seller discovery.")
    cs.add_argument("intent")
    cs.add_argument("--max-price", type=int, default=None, dest="max_price")
    cs.add_argument("--category", default=None)
    cs.add_argument("--k", type=int, default=10)
    cs.set_defaults(func=_cmd_commerce_search)

    crs = sub.add_parser("commerce-register-seller", help="Register a seller (free).")
    crs.add_argument("name")
    crs.add_argument("checkout_session_url")
    crs.set_defaults(func=_cmd_commerce_register_seller)

    cal = sub.add_parser("commerce-add-listings", help="Add listings to a seller (free).")
    cal.add_argument("seller_id", type=int)
    cal.add_argument("listings_json", help='JSON array, e.g. \'[{"item_id":"sku-1","name":"Boots","description":"..."}]\'')
    cal.set_defaults(func=_cmd_commerce_add_listings)

    cf = sub.add_parser("commerce-feedback", help="Report a real outcome as the BUYER (purchased/not_purchased/irrelevant).")
    cf.add_argument("seller_id", type=int)
    cf.add_argument("item_id")
    cf.add_argument("outcome", choices=["purchased", "not_purchased", "irrelevant"])
    cf.add_argument("--match-id", default=None, dest="match_id", help="From a real commerce-search result -- required for this to count toward your reputation tier.")
    cf.set_defaults(func=_cmd_commerce_feedback)

    csf = sub.add_parser("commerce-seller-feedback", help="Report a real outcome as the SELLER (fulfilled/buyer_never_completed/disputed).")
    csf.add_argument("match_id")
    csf.add_argument("outcome", choices=["fulfilled", "buyer_never_completed", "disputed"])
    csf.add_argument("--rating", type=int, default=None, choices=[1, 2, 3, 4, 5])
    csf.set_defaults(func=_cmd_commerce_seller_feedback)

    rep = sub.add_parser("commerce-reputation", help="Check your own key's real reputation tier.")
    rep.set_defaults(func=_cmd_commerce_reputation)

    vm = sub.add_parser("commerce-verify-match", help="As a seller: check a buyer's reputation tier for one of your matches, without seeing their identity.")
    vm.add_argument("match_id")
    vm.set_defaults(func=_cmd_commerce_verify_match)

    ns = sub.add_parser("commerce-network-stats", help="Public aggregate stats for the whole trust network.")
    ns.set_defaults(func=_cmd_commerce_network_stats)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
