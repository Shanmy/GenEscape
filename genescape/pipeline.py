"""
GenEscape Pipeline — implements Algorithm 1 from the paper:
Hierarchical Puzzle Optimization with Multi-Agent Feedback.

Stages:
  0 — Designer creates initial scene graph G0 and ground-truth solution S
  1 — GRAPH: player-examiner loop refines scene graph
  2 — LAYOUT: builder creates layout, player-examiner loop refines layout
  3 — IMAGE: builder generates image, player-examiner loop refines image
"""

import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from genescape.llm_provider import BaseLLMProvider
from genescape.agents.designer import DesignerAgent
from genescape.agents.player import PlayerAgent
from genescape.agents.examiner import ExaminerAgent
from genescape.agents.builder import BuilderAgent


def _banner(text: str) -> None:
    """Print a prominent stage banner."""
    width = 60
    border = "=" * width
    print(f"\n{border}")
    print(f"  {text}")
    print(f"{border}")


def _log(label: str, msg: str) -> None:
    """Print a labeled log line."""
    print(f"  [{label}] {msg}")


def _timestamp() -> str:
    """Return a sortable timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class GenEscapePipeline:
    """
    Orchestrates the full multi-agent escape room generation pipeline.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        output_dir: str = "output",
        max_iters: int = 5,
    ) -> None:
        self.provider = provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_iters = max_iters

        # Instantiate agents
        self.designer = DesignerAgent(provider)
        self.player = PlayerAgent(provider)
        self.examiner = ExaminerAgent(provider)
        self.builder = BuilderAgent(provider)

    def run(
        self,
        scene_keyword: str,
        objects: list[str],
        solution_length: int = 5,
    ) -> dict:
        """
        Execute the full GenEscape algorithm.

        Args:
            scene_keyword: Theme/context for the room (e.g., "classroom")
            objects: List of key objects to include (e.g., ["ladder", "key"])
            solution_length: Target number of steps in the solution

        Returns:
            dict with keys:
                scene_graph, layout, image_path, solution, description, iterations
        """
        ts = _timestamp()
        iterations = {"graph": 0, "layout": 0, "image": 0}

        # Create a per-run subfolder: e.g. output/prison_bucket_blanket_20260316_143022/
        slug_parts = [scene_keyword] + objects
        slug = "_".join(re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_") for p in slug_parts)
        run_dir = self.output_dir / f"{slug}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _log("Output", f"Run folder: {run_dir}")

        # ===================================================================
        _banner("Stage 0: Initial Design")
        # ===================================================================
        _log("Designer", f"Generating scene for keyword='{scene_keyword}', "
                          f"objects={objects}, solution_length={solution_length}")

        design = self.designer.design(scene_keyword, objects, solution_length)
        description: str = design["description"]
        scene_graph: str = design["scene_graph"]
        solution: list[str] = design["solution"]

        _log("Designer", f"Scene graph generated ({len(scene_graph)} chars)")
        _log("Designer", f"Solution has {len(solution)} steps")
        for i, step in enumerate(solution, 1):
            _log("Solution", f"  {i}. {step}")

        # Save initial design
        self._save_text(scene_graph, "stage0_scene_graph.yaml", run_dir)
        self._save_text(description, "stage0_description.txt", run_dir)
        self._save_solution(solution, "stage0_solution.txt", run_dir)

        # ===================================================================
        _banner("Stage 1: Scene Graph Optimization (GRAPH)")
        # ===================================================================
        for iteration in range(1, self.max_iters + 1):
            _log("Stage 1", f"Iteration {iteration}/{self.max_iters}")

            player_solution = self.player.solve_from_graph(scene_graph, description)
            _log("Player", f"Proposed {len(player_solution)}-step solution")

            feedback = self.examiner.check(solution, player_solution)
            iterations["graph"] += 1

            if not feedback:
                _log("Examiner", "Solutions match — scene graph approved!")
                break

            _log("Examiner", f"Found {len(feedback)} issue(s):")
            for item in feedback:
                _log("  >", item)

            if iteration < self.max_iters:
                _log("Examiner", "Refining scene graph...")
                scene_graph = self.examiner.refine_graph(scene_graph, feedback)
                self._save_text(scene_graph, f"stage1_iter{iteration}_scene_graph.yaml", run_dir)
            else:
                _log("Stage 1", "Max iterations reached — proceeding with current graph.")

        self._save_text(scene_graph, "stage1_final_scene_graph.yaml", run_dir)

        # ===================================================================
        _banner("Stage 2: Layout Optimization (LAYOUT)")
        # ===================================================================
        _log("Builder", "Creating initial 2D layout from scene graph...")
        layout: str = self.builder.create_layout(scene_graph, description)
        self._save_text(layout, "stage2_initial_layout.yaml", run_dir)
        _log("Builder", f"Layout created ({len(layout)} chars)")

        for iteration in range(1, self.max_iters + 1):
            _log("Stage 2", f"Iteration {iteration}/{self.max_iters}")

            player_solution = self.player.solve_from_layout(layout, description)
            _log("Player", f"Proposed {len(player_solution)}-step solution from layout")

            feedback = self.examiner.check(solution, player_solution)
            iterations["layout"] += 1

            if not feedback:
                _log("Examiner", "Solutions match — layout approved!")
                break

            _log("Examiner", f"Found {len(feedback)} issue(s):")
            for item in feedback:
                _log("  >", item)

            if iteration < self.max_iters:
                _log("Builder", "Refining layout...")
                layout = self.builder.refine_layout(layout, feedback, scene_graph)
                self._save_text(layout, f"stage2_iter{iteration}_layout.yaml", run_dir)
            else:
                _log("Stage 2", "Max iterations reached — proceeding with current layout.")

        self._save_text(layout, "stage2_final_layout.yaml", run_dir)

        # ===================================================================
        _banner("Stage 3: Image Generation and Optimization (IMAGE)")
        # ===================================================================
        image_path_base = str(run_dir / "stage3_room_v0.png")
        _log("Builder", "Generating initial room image...")

        try:
            image_path = self.builder.generate_image(layout, description, scene_graph, image_path_base)
            _log("Builder", f"Image saved: {image_path}")
        except Exception as e:
            _log("Builder", f"Image generation error: {e}")
            traceback.print_exc()
            image_path = image_path_base  # Will be missing; handled downstream

        for iteration in range(1, self.max_iters + 1):
            _log("Stage 3", f"Iteration {iteration}/{self.max_iters}")

            # Check if image is a real image or fallback description
            if not Path(image_path).exists() or image_path.endswith("_description.txt"):
                _log("Player", "No real image available; skipping vision solve — using layout solve.")
                player_solution = self.player.solve_from_layout(layout, description)
            else:
                try:
                    player_solution = self.player.solve_from_image(image_path, description)
                    _log("Player", f"Proposed {len(player_solution)}-step solution from image")
                except Exception as e:
                    _log("Player", f"Vision solve failed ({e}); falling back to layout solve.")
                    player_solution = self.player.solve_from_layout(layout, description)

            feedback = self.examiner.check(solution, player_solution)
            iterations["image"] += 1

            if not feedback:
                _log("Examiner", "Solutions match — image approved!")
                break

            _log("Examiner", f"Found {len(feedback)} issue(s):")
            for item in feedback:
                _log("  >", item)

            if iteration < self.max_iters:
                refined_path = str(run_dir / f"stage3_room_v{iteration}.png")
                _log("Builder", f"Refining image (attempt {iteration})...")
                try:
                    image_path = self.builder.refine_image(
                        image_path, layout, feedback, description, refined_path
                    )
                    _log("Builder", f"Refined image saved: {image_path}")
                except Exception as e:
                    _log("Builder", f"Image refinement error: {e}")
                    traceback.print_exc()
                    break
            else:
                _log("Stage 3", "Max iterations reached — using current image.")

        # ===================================================================
        _banner("Pipeline Complete")
        # ===================================================================
        _log("Result", f"Run folder:  {run_dir}")
        _log("Result", f"Final image: {image_path}")
        _log("Result", f"Total graph iterations: {iterations['graph']}")
        _log("Result", f"Total layout iterations: {iterations['layout']}")
        _log("Result", f"Total image iterations: {iterations['image']}")

        # Save final solution for use by human interface
        final_solution_path = str(run_dir / "final_solution.txt")
        self._save_solution(solution, "final_solution.txt", run_dir)

        return {
            "scene_graph": scene_graph,
            "layout": layout,
            "image_path": image_path,
            "solution": solution,
            "description": description,
            "iterations": iterations,
            "solution_path": final_solution_path,
            "run_dir": str(run_dir),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_text(self, content: str, filename: str, directory: Optional[Path] = None) -> Path:
        """Save a text file to directory (defaults to output_dir)."""
        path = (directory or self.output_dir) / filename
        path.write_text(content, encoding="utf-8")
        return path

    def _save_solution(self, solution: list[str], filename: str, directory: Optional[Path] = None) -> Path:
        """Save the solution as a numbered text file."""
        content = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(solution))
        return self._save_text(content, filename, directory)
