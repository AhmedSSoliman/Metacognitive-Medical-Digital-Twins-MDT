#!/usr/bin/env python3
"""
scripts/clinical_review_ui.py

A simple CLI tool to perform the clinical face-validity review of candidate
hyperedges mined in Phase 3. It reads a hypergraph JSON file, prompts the user
to accept or reject each edge, and saves the result with the status updated to
"CLINICALLY_REVIEWED".

Resume support (added 2026-08-14): at 4187 edges, one uninterrupted sitting
isn't realistic. Progress now autosaves to input_file's own path every
--autosave_every edges (default 25), and on restart any edge that already
has an "approved" key (True or False) is skipped -- so re-running the same
command just continues where you left off. Type 'q' at any prompt to save
immediately and exit cleanly (equivalent to Ctrl-C but without losing the
in-progress edge's un-answered state).
"""

import json
import argparse
import sys
from pathlib import Path

def _save(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Review and approve candidate hyperedges.")
    parser.add_argument("input_file", help="Path to the derived_hypergraph JSON file")
    parser.add_argument("--reviewer", default="IC3_Collaborator", help="Name of the reviewer")
    parser.add_argument("--autosave_every", type=int, default=25,
                         help="Autosave progress back to input_file every N reviewed edges.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File {args.input_file} not found.")
        sys.exit(1)

    with open(input_path, 'r') as f:
        data = json.load(f)

    if "hyperedges" not in data:
        print("Error: Invalid format, no 'hyperedges' found.")
        sys.exit(1)

    hyperedges = data["hyperedges"]
    total = len(hyperedges)
    already_done = sum(1 for e in hyperedges if "approved" in e)
    if already_done:
        print(f"Resuming: {already_done}/{total} edges already reviewed in a prior session, skipping those.")
    print(f"Loaded {total} candidate hyperedges for review from {input_path.name}")
    print("-" * 50)

    # Note: we are not using the FaceValidityReview class directly because it saves
    # to a separate tracker file. We will modify the main JSON's hyperedges list
    # by adding an 'approved' flag to each edge to make it self-contained for Phase 4.

    approved_count = sum(1 for e in hyperedges if e.get("approved") is True)
    reviewed_this_session = 0
    quit_early = False

    for i, edge in enumerate(hyperedges):
        if "approved" in edge:
            continue  # already reviewed in a prior session

        variables = ", ".join(edge["variables"])
        p_val = edge.get("p_value", "N/A")
        print(f"Edge {i+1}/{total}: {{ {variables} }} (p-val: {p_val})")

        while True:
            resp = input(f"Approve this hyperedge? [y/N/q=save&quit]: ").strip().lower()
            if resp in ['y', 'yes']:
                edge["approved"] = True
                edge["reviewer"] = args.reviewer
                approved_count += 1
                reviewed_this_session += 1
                break
            elif resp in ['n', 'no', '']:
                edge["approved"] = False
                edge["reviewer"] = args.reviewer
                reviewed_this_session += 1
                break
            elif resp in ['q', 'quit']:
                quit_early = True
                break
            else:
                print("Please enter 'y', 'n', or 'q'.")
        print("-" * 50)

        if quit_early:
            break

        if args.autosave_every > 0 and reviewed_this_session % args.autosave_every == 0:
            _save(data, input_path)
            done_so_far = sum(1 for e in hyperedges if "approved" in e)
            print(f"[autosaved progress: {done_so_far}/{total} edges reviewed so far]")

    if quit_early:
        _save(data, input_path)
        done_so_far = sum(1 for e in hyperedges if "approved" in e)
        print(f"\nSaved progress ({done_so_far}/{total} reviewed) to {input_path}.")
        print(f"Re-run the same command to resume from where you left off.")
        return

    # All edges now have an "approved" key (either from this session or a prior one).
    reviewed_total = sum(1 for e in hyperedges if "approved" in e)
    if reviewed_total < total:
        # Shouldn't happen (loop only exits early via quit_early), but guard anyway.
        _save(data, input_path)
        print(f"\n{reviewed_total}/{total} reviewed so far; re-run to continue.")
        return

    print(f"\nReview complete! Approved {approved_count}/{total} hyperedges.")

    kept_edges = [e for e in hyperedges if e.get("approved")]
    output_path = input_path.parent / "derived_hypergraph.json"

    save_resp = input(f"Save final reviewed hypergraph ({len(kept_edges)} approved edges) to {output_path}? [Y/n]: ").strip().lower()
    if save_resp not in ['n', 'no']:
        final_data = dict(data)
        final_data["hyperedges"] = kept_edges
        final_data["status"] = "CLINICALLY_REVIEWED"
        _save(final_data, output_path)
        print(f"Successfully saved to {output_path}.")
        print("You can now proceed to Phase 4.")
    else:
        print("Save aborted. Your per-edge progress is still saved in the original input file.")

if __name__ == "__main__":
    main()
