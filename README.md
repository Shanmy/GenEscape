# GenEscape: Hierarchical Multi-Agent Generation of Escape Room Puzzles 🧩

A Python re-implementation of the [GenEscape paper](https://arxiv.org/abs/2506.21839) by **Mengyi Shan, Brian Curless, Ira Kemelmacher-Shlizerman, and Steve Seitz** (University of Washington).
Published at the **Hi-Gen Workshop, ICCV 2025**.

GenEscape uses a hierarchical multi-agent framework to generate 2D escape room puzzle images that are both **visually appealing** and **logically solvable**. Four specialized LLM agents collaborate through iterative feedback loops — refining a scene graph, a 2D layout, and a final photorealistic image — until the puzzle is verified as solvable without shortcuts.

---

## How It Works

The system implements **Algorithm 1** from the paper across three refinement stages:

```
Input: scene keyword (e.g. "prison"), objects (e.g. ["bucket", "blanket"])

Stage 0 — Designer creates initial scene graph G₀ and ground-truth solution S
Stage 1 — GRAPH:  Player solves from graph → Examiner checks → Examiner refines graph   (repeat)
Stage 2 — LAYOUT: Builder creates layout  → Player solves → Builder refines layout      (repeat)
Stage 3 — IMAGE:  Builder generates image → Player solves from image → Builder refines  (repeat)

Output: photorealistic escape room image + verified solution
```

### The Four Agents

| Agent | Role |
|-------|------|
| **Designer** | Generates the scene description, YAML scene graph, and ground-truth solution |
| **Player** | Simulates a human solver; proposes action sequences from graph, layout, or image |
| **Examiner** | Compares Player's solution to ground truth; identifies shortcuts and missing steps |
| **Builder** | Creates 2D spatial layouts and generates photorealistic room images |

---

## Installation

```bash
git clone https://github.com/your-username/GenEscape.git
cd GenEscape
pip install -r requirements.txt
```

**Requirements:** Python 3.10+

---

## Configuration

Set your API key as an environment variable depending on which provider you use:

```bash
# For OpenAI (GPT-4o + DALL-E 3)
export OPENAI_API_KEY=your_openai_api_key

# For Google Gemini (gemini-2.5-flash + gemini-2.5-flash-image)
export GOOGLE_API_KEY=your_google_api_key
```

### Optional environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | Default provider (`openai` or `gemini`) |
| `OPENAI_TEXT_MODEL` | `gpt-4o` | OpenAI model for text/vision |
| `OPENAI_IMAGE_MODEL` | `dall-e-3` | OpenAI model for image generation |
| `GEMINI_TEXT_MODEL` | `gemini-2.5-flash` | Gemini model for text/vision |
| `GEMINI_IMAGE_MODEL` | `gemini-2.5-flash-image` | Gemini model for image generation |
| `MAX_REFINEMENT_ITERS` | `5` | Max refinement iterations per stage |
| `OUTPUT_DIR` | `output` | Root directory for saved outputs |

---

## Usage

### Generate a puzzle

```bash
# Using OpenAI
python main.py --provider openai --scene classroom --objects ladder key

# Using Gemini
python main.py --provider gemini --scene prison --objects bucket blanket

# Customize solution length and refinement iterations
python main.py --scene laboratory --objects magnet key --solution-length 6 --max-iters 3
```

### Generate and immediately play

```bash
python main.py --provider gemini --scene "birthday party" --objects balloon dart --play
```

### All options

```
--provider   openai | gemini         LLM provider (default: openai)
--scene      SCENE                   Scene keyword / theme
--objects    OBJ [OBJ ...]           Key objects to include
--solution-length  N                 Target number of solution steps (default: 5)
--max-iters  N                       Max refinement iterations per stage (default: 5)
--output-dir DIR                     Root output directory (default: output/)
--play                               Enter interactive play mode after generation
--image      PATH                    Existing image for play-only mode
--solution   PATH                    Existing solution file for play-only mode
```

---

## Output Structure

Each run creates a timestamped subfolder inside `output/`:

```
output/
  prison_bucket_blanket_20260316_112549/
    stage0_scene_graph.yaml       # Initial scene graph from Designer
    stage0_description.txt        # Room description
    stage0_solution.txt           # Ground-truth solution
    stage1_final_scene_graph.yaml # Scene graph after refinement
    stage2_initial_layout.yaml    # First 2D layout from Builder
    stage2_final_layout.yaml      # Layout after refinement
    stage3_room_v0.png            # Generated room image
    stage3_room_v1.png            # Refined image (if needed)
    final_solution.txt            # Final verified solution
```

---

## Human Interface

When `--play` is active, you interact with the generated puzzle in a conversational loop. An AI Examiner — equipped with the ground-truth solution — evaluates each action and guides you toward the correct path without giving it away directly.

```
You > I grab the key from the shelf
AI  > The key is on a high shelf, out of reach from the floor. What do you try next?

You > I move the bucket under the shelf and stand on it
AI  > You step onto the bucket. It wobbles but holds. You can now reach the shelf.

You > hint
AI  > Look carefully at the objects on the table. One of them might help you with the lock.
```

Special commands: `hint`, `look` (describe the room), `solution` (reveal answer), `quit`.

---

## Project Structure

```
GenEscape/
├── main.py                        # CLI entry point
├── requirements.txt
├── diagnose.py                    # API diagnostics (check available models)
└── genescape/
    ├── config.py                  # Settings and environment variables
    ├── llm_provider.py            # OpenAI and Gemini provider abstraction
    ├── pipeline.py                # Algorithm 1 — three-stage refinement loop
    ├── human_interface.py         # Interactive puzzle-playing mode
    └── agents/
        ├── designer.py            # Designer agent
        ├── player.py              # Player agent
        ├── examiner.py            # Examiner agent
        └── builder.py             # Builder agent (layout + image generation)
```

---

## Supported Models

### OpenAI
| Task | Model |
|------|-------|
| Text & vision | `gpt-4o` |
| Image generation | `dall-e-3` |

### Google Gemini
| Task | Model |
|------|-------|
| Text & vision | `gemini-2.5-flash` |
| Image generation | `gemini-2.5-flash-image` (or `imagen-4.0-generate-001`) |

To use a different model, set the corresponding environment variable (e.g. `GEMINI_TEXT_MODEL=gemini-2.5-pro`).

---

## Citation

```bibtex
@inproceedings{shan2025genescape,
  title     = {GenEscape: Hierarchical Multi-Agent Generation of Escape Room Puzzles},
  author    = {Shan, Mengyi and Curless, Brian and Kemelmacher-Shlizerman, Ira and Seitz, Steve},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  series    = {Hi-Gen Workshop},
  year      = {2025}
}
```
