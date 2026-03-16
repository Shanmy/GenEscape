"""
GenEscape CLI entry point.

Implements: Hierarchical Multi-Agent Generation of Escape Room Puzzles

Usage:
    python main.py [--provider openai|gemini] [--scene SCENE]
                   [--objects OBJ1 OBJ2 ...] [--solution-length N]
                   [--play] [--image PATH] [--output-dir DIR]

Examples:
    # Generate a classroom escape room with OpenAI
    python main.py --provider openai --scene classroom --objects ladder key --solution-length 5

    # Generate with Gemini
    python main.py --provider gemini --scene "abandoned laboratory" --objects "test tube" "key card" lever

    # Play an existing puzzle
    python main.py --play --image output/room.png --solution output/solution.txt

    # Generate and immediately play
    python main.py --scene library --play
"""

import argparse
import os
import sys
from pathlib import Path


def _check_api_key(provider_name: str) -> bool:
    """Verify that the required API key is set; print helpful message if not."""
    if provider_name == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            print("\n[Error] OPENAI_API_KEY is not set.")
            print("  To fix: export OPENAI_API_KEY=your_openai_api_key")
            print("  Get a key at: https://platform.openai.com/api-keys")
            return False
    elif provider_name == "gemini":
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            print("\n[Error] GOOGLE_API_KEY is not set.")
            print("  To fix: export GOOGLE_API_KEY=your_google_api_key")
            print("  Get a key at: https://aistudio.google.com/app/apikey")
            return False
    return True


def _load_solution_from_file(solution_path: str) -> list[str]:
    """Load a numbered solution from a text file."""
    path = Path(solution_path)
    if not path.exists():
        print(f"[Error] Solution file not found: {solution_path}")
        sys.exit(1)

    steps: list[str] = []
    import re
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading number
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned:
            steps.append(cleaned)
    return steps


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="genescape",
        description="GenEscape: Hierarchical Multi-Agent Generation of Escape Room Puzzles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Provider
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default=os.environ.get("LLM_PROVIDER", "openai").lower(),
        help="LLM provider to use (default: openai, or LLM_PROVIDER env var)",
    )

    # Scene parameters
    parser.add_argument(
        "--scene",
        default="classroom",
        metavar="SCENE",
        help="Scene keyword / theme for the escape room (default: classroom)",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["ladder", "key"],
        metavar="OBJECT",
        help="Objects to include in the puzzle (default: ladder key)",
    )
    parser.add_argument(
        "--solution-length",
        type=int,
        default=5,
        dest="solution_length",
        metavar="N",
        help="Target number of solution steps (default: 5)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "output"),
        dest="output_dir",
        metavar="DIR",
        help="Directory for saving outputs (default: output)",
    )

    # Refinement
    parser.add_argument(
        "--max-iters",
        type=int,
        default=int(os.environ.get("MAX_REFINEMENT_ITERS", "5")),
        dest="max_iters",
        metavar="N",
        help="Maximum refinement iterations per stage (default: 5)",
    )

    # Human play mode
    parser.add_argument(
        "--play",
        action="store_true",
        help="Enter interactive human play mode after generation (or standalone)",
    )
    parser.add_argument(
        "--image",
        default=None,
        metavar="PATH",
        help="Path to existing room image (for --play without generation)",
    )
    parser.add_argument(
        "--solution",
        default=None,
        metavar="PATH",
        help="Path to existing solution file (for --play with --image)",
    )
    parser.add_argument(
        "--description",
        default=None,
        metavar="TEXT",
        help="Scene description (for --play with --image, if no description file)",
    )

    return parser


def main() -> None:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Validate provider choice
    provider_name = args.provider.lower()
    if not _check_api_key(provider_name):
        sys.exit(1)

    # Import after key check so missing packages show clean errors
    try:
        from genescape.config import LLMProvider
        from genescape.llm_provider import get_provider
    except ImportError as e:
        print(f"\n[Error] Import failed: {e}")
        print("  Run: pip install -r requirements.txt")
        sys.exit(1)

    # Build provider
    provider_enum = LLMProvider.GEMINI if provider_name == "gemini" else LLMProvider.OPENAI
    try:
        provider = get_provider(provider_enum)
    except Exception as e:
        print(f"\n[Error] Failed to initialize {provider_name} provider: {e}")
        sys.exit(1)

    print(f"\n  GenEscape — using provider: {provider_name.upper()}")
    print(f"  Output directory: {args.output_dir}")

    # ----------------------------------------------------------------
    # Mode A: Play an existing puzzle (no generation)
    # ----------------------------------------------------------------
    if args.play and args.image:
        print("\n  [Play Mode] Loading existing puzzle...")

        image_path = args.image
        if not Path(image_path).exists():
            print(f"[Error] Image file not found: {image_path}")
            sys.exit(1)

        # Load solution
        if args.solution:
            solution = _load_solution_from_file(args.solution)
        else:
            print("[Warning] No --solution file provided. Using a placeholder solution.")
            solution = ["Explore the room", "Find all items", "Escape through the door"]

        # Load description
        description = args.description or "An escape room. Find a way out!"

        # Check for a description file alongside the image
        desc_path = Path(image_path).with_suffix(".description.txt")
        if desc_path.exists():
            description = desc_path.read_text(encoding="utf-8").strip()

        from genescape.human_interface import HumanInterface
        interface = HumanInterface(provider)
        interface.play(image_path, description, solution)
        return

    # ----------------------------------------------------------------
    # Mode B: Generate puzzle (with optional play after)
    # ----------------------------------------------------------------
    print(f"\n  Scene: {args.scene}")
    print(f"  Objects: {args.objects}")
    print(f"  Solution length: {args.solution_length}")
    print(f"  Max refinement iterations: {args.max_iters}")
    print()

    # Validate solution length
    if args.solution_length < 2:
        print("[Error] --solution-length must be at least 2.")
        sys.exit(1)

    from genescape.pipeline import GenEscapePipeline

    pipeline = GenEscapePipeline(
        provider=provider,
        output_dir=args.output_dir,
        max_iters=args.max_iters,
    )

    try:
        result = pipeline.run(
            scene_keyword=args.scene,
            objects=args.objects,
            solution_length=args.solution_length,
        )
    except KeyboardInterrupt:
        print("\n\n  Generation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("  GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Run folder:  {result.get('run_dir', args.output_dir)}")
    print(f"  Image:       {result['image_path']}")
    print(f"  Solution:    {result.get('solution_path', 'see run folder')}")
    print()
    print("  Ground-truth solution:")
    for i, step in enumerate(result["solution"], 1):
        print(f"    {i}. {step}")
    print()

    # Optionally enter play mode
    if args.play:
        from genescape.human_interface import HumanInterface
        print("  Entering interactive play mode...")
        interface = HumanInterface(provider)
        interface.play(
            image_path=result["image_path"],
            description=result["description"],
            official_solution=result["solution"],
        )


if __name__ == "__main__":
    main()
