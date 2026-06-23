"""Generate conceptual SVG/PNG figure drafts for E024.

These figures are intentionally schematic and editable. They are designed as
high-clarity inputs for downstream image refinement rather than as final
publication artwork.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLE_DIR = (
    REPO_ROOT
    / "article"
    / "Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking"
)
OUTPUT_DIR = ARTICLE_DIR / "figures3_concepts"


COLORS = {
    "navy": "#1f3b73",
    "blue": "#2f7de1",
    "blue_soft": "#eaf3ff",
    "teal": "#1f9e8a",
    "teal_soft": "#e8fbf7",
    "green": "#34a853",
    "green_soft": "#ecf9ef",
    "orange": "#ef8a17",
    "orange_soft": "#fff3e5",
    "purple": "#7b61c9",
    "purple_soft": "#f2ecff",
    "red": "#d9534f",
    "red_soft": "#fdeeee",
    "gray": "#64748b",
    "gray_soft": "#f5f7fb",
    "ink": "#18212f",
    "gold": "#c7921b",
}


def add_round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str = "white",
    ec: str = "#d6deeb",
    lw: float = 1.4,
    radius: float = 2.5,
    zorder: int = 1,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.4,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def add_title(fig, ax, title: str, subtitle: str | None = None) -> None:
    ax.text(
        50,
        96.5,
        title,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=COLORS["navy"],
    )
    if subtitle:
        ax.text(
            50,
            92.7,
            subtitle,
            ha="center",
            va="center",
            fontsize=10.5,
            color=COLORS["gray"],
        )


def add_section_header(ax, x: float, y: float, w: float, text: str, color: str) -> None:
    add_round_box(ax, x, y, w, 6.0, fc=color, ec=color, lw=0.0, radius=2.0, zorder=3)
    ax.text(
        x + w / 2,
        y + 3.0,
        text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
        zorder=4,
    )


def add_icon_circle(ax, x: float, y: float, label: str, color: str, soft: str) -> None:
    ax.add_patch(Circle((x, y), 3.0, facecolor=soft, edgecolor=color, linewidth=1.2))
    ax.text(x, y, label, ha="center", va="center", fontsize=12, fontweight="bold", color=color)


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#8593a6",
    lw: float = 2.0,
    style: str = "-|>",
    rad: float = 0.0,
    linestyle: str = "-",
    mutation_scale: int = 14,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
    )
    ax.add_patch(arrow)
    return arrow


def setup_canvas() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def draw_branch_schematic(ax, x: float, y: float, scale: float = 1.0, highlight: bool = False) -> None:
    trunk_y = y
    ax.plot(
        [x - 14 * scale, x + 14 * scale],
        [trunk_y, trunk_y],
        color="#79a7e3",
        linewidth=5 * scale,
        solid_capstyle="round",
        zorder=2,
    )
    branch_xs = [x - 10 * scale, x - 3 * scale, x + 4 * scale, x + 11 * scale]
    branch_cols = ["#8fb8f1", "#f4a340" if highlight else "#8fb8f1", "#8fb8f1", "#8fb8f1"]
    labels = ["b0", "b1", "b2", "b3"]
    for bx, col, label in zip(branch_xs, branch_cols, labels):
        ax.plot([bx, bx], [trunk_y, trunk_y + 9 * scale], color="#7aa2dd", linewidth=3 * scale)
        ax.add_patch(Circle((bx, trunk_y + 11 * scale), 1.7 * scale, facecolor=col, edgecolor="#4f78b3", linewidth=1))
        ax.text(bx, trunk_y + 15 * scale, label, ha="center", va="center", fontsize=8 * scale, color=COLORS["ink"])


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def figure_biological_motivation() -> None:
    fig, ax = setup_canvas()
    add_title(
        fig,
        ax,
        "Biological motivation for a slow accessibility variable",
        "A schematic bridge from branch-local biology to the latent simulator variable M_b",
    )

    add_round_box(ax, 4, 10, 26, 76, fc=COLORS["purple_soft"], ec="#d9cef7")
    add_section_header(ax, 7, 82, 20, "A. Fast access now", COLORS["purple"])
    add_icon_circle(ax, 10, 70, "1", COLORS["purple"], "#faf6ff")
    ax.text(14, 72, "cue input", fontsize=13, fontweight="bold", color=COLORS["ink"], va="center")
    ax.text(14, 66, "momentary opening of active branches", fontsize=10.5, color=COLORS["gray"], va="center")
    add_icon_circle(ax, 10, 55, "2", COLORS["purple"], "#faf6ff")
    ax.text(14, 57, "context and local state", fontsize=13, fontweight="bold", color=COLORS["ink"], va="center")
    ax.text(14, 51, "fast disambiguation without persistence", fontsize=10.5, color=COLORS["gray"], va="center")
    add_icon_circle(ax, 10, 40, "3", COLORS["purple"], "#faf6ff")
    ax.text(14, 42, "selects branches for this trial", fontsize=13, fontweight="bold", color=COLORS["ink"], va="center")
    ax.text(14, 36, "does not by itself write a durable bias", fontsize=10.5, color=COLORS["gray"], va="center")

    add_round_box(ax, 34, 18, 30, 60, fc=COLORS["green_soft"], ec="#cfead8")
    add_section_header(ax, 38, 74, 22, "B. Slow branch-local processes", COLORS["green"])
    process_rows = [
        ("spine geometry", "neck state, local compartmentalization"),
        ("actin / microtubules", "branch-local structural remodeling"),
        ("translation / energy", "transport readiness, mitochondrial support"),
    ]
    y0 = 62
    for idx, (line1, line2) in enumerate(process_rows):
        yy = y0 - idx * 15
        add_icon_circle(ax, 39, yy, chr(ord("A") + idx), COLORS["green"], "#f8fffb")
        ax.text(43, yy + 1.5, line1, fontsize=12.5, fontweight="bold", color=COLORS["ink"], va="center")
        ax.text(43, yy - 3.5, line2, fontsize=10.2, color=COLORS["gray"], va="center")

    ax.add_patch(Circle((76, 50), 10.5, facecolor=COLORS["orange_soft"], edgecolor=COLORS["orange"], linewidth=2.2))
    ax.text(76, 55.5, r"$M_b$", ha="center", va="center", fontsize=26, fontweight="bold", color=COLORS["orange"])
    ax.text(
        76,
        47.5,
        "slow structural\naccessibility bias",
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["ink"],
    )
    ax.text(76, 37.5, "phenomenological, not\none molecule", ha="center", va="center", fontsize=9.8, color=COLORS["gray"])

    add_arrow(ax, (30, 50), (65, 50), color=COLORS["purple"], lw=2.6)
    ax.text(47, 53.2, "separate timescale", ha="center", fontsize=10.5, color=COLORS["purple"])
    for yy in (62, 47, 32):
        add_arrow(ax, (64, yy), (66.2, 52), color=COLORS["green"], lw=2.2, rad=0.05)

    add_round_box(ax, 71, 12, 24, 18, fc=COLORS["blue_soft"], ec="#c8dcfb")
    ax.text(83, 24, "Later effects", ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["navy"])
    ax.text(
        83,
        16.8,
        "biases later reuse,\nlinking, damage sensitivity,\nand rescue selectivity",
        ha="center",
        va="center",
        fontsize=10.3,
        color=COLORS["ink"],
    )
    add_arrow(ax, (76, 39.5), (83, 30.2), color=COLORS["orange"], lw=2.6)

    save_figure(fig, "Fig_e024_01_biological_motivation")


def figure_comparator_fairness() -> None:
    fig, ax = setup_canvas()
    add_title(
        fig,
        ax,
        "Comparator fairness and model-discrimination logic",
        "A schematic showing why alternatives are tested on both output-level and mechanistic criteria",
    )

    add_round_box(ax, 4, 12, 22, 74, fc=COLORS["gray_soft"], ec="#d5deea")
    add_section_header(ax, 7, 82, 16, "A. Tested alternatives", COLORS["blue"])
    comparators = [
        ("fast gating only", COLORS["purple_soft"], COLORS["purple"]),
        ("fixed overlap", COLORS["blue_soft"], COLORS["blue"]),
        ("weight only", COLORS["orange_soft"], COLORS["orange"]),
        ("global gain", COLORS["red_soft"], COLORS["red"]),
        ("shuffled replay", COLORS["teal_soft"], COLORS["teal"]),
    ]
    for idx, (label, soft, color) in enumerate(comparators):
        yy = 70 - idx * 12
        add_round_box(ax, 7, yy - 4, 16, 8, fc=soft, ec=color)
        ax.text(15, yy, label, ha="center", va="center", fontsize=11.5, fontweight="bold", color=COLORS["ink"])

    add_round_box(ax, 31, 14, 28, 70, fc=COLORS["blue_soft"], ec="#c7daf8")
    add_section_header(ax, 35, 80, 20, "B. Fair test", COLORS["navy"])
    steps = [
        "1. Match output-level linking?",
        "2. Match damage pattern?",
        "3. Match targeted rescue?",
        "4. Match overlap writing + specificity?",
    ]
    for idx, step in enumerate(steps):
        yy = 69 - idx * 13.5
        add_round_box(ax, 36, yy - 4.5, 18, 9, fc="white", ec="#a8bfeb", radius=2.0)
        ax.text(45, yy, step, ha="center", va="center", fontsize=10.8, color=COLORS["ink"], wrap=True)
        if idx < len(steps) - 1:
            add_arrow(ax, (45, yy - 5.4), (45, yy - 9.5), color=COLORS["navy"], lw=1.8)

    add_round_box(ax, 63, 14, 33, 70, fc=COLORS["green_soft"], ec="#cbe8d3")
    add_section_header(ax, 67, 80, 25, "C. Reading the outcome", COLORS["green"])
    ax.text(79.5, 70, "Partial matches can be real", ha="center", fontsize=14, fontweight="bold", color=COLORS["green"])
    ax.text(
        79.5,
        61,
        "weight-only or gain-based models may\nmatch some output-level effects",
        ha="center",
        fontsize=11,
        color=COLORS["ink"],
    )
    add_round_box(ax, 67, 45, 25, 12, fc="white", ec="#9ad2aa")
    ax.text(79.5, 51, "But the full claim needs:", ha="center", fontsize=12, fontweight="bold", color=COLORS["navy"])
    ax.text(
        79.5,
        37,
        "overlap writing\n+ linking gain\n+ rescue selectivity\n+ motif specificity",
        ha="center",
        fontsize=12,
        color=COLORS["ink"],
    )
    add_round_box(ax, 68, 18, 23, 11, fc="#ffffff", ec=COLORS["green"], lw=2.0)
    ax.text(79.5, 23.5, "Only full model\npasses the full stack", ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["green"])

    for src_y in (70, 58, 46, 34, 22):
        add_arrow(ax, (23.2, src_y), (35.5, 69), color="#9ba8b7", lw=1.4, rad=0.12 - (src_y - 22) / 240)
    add_arrow(ax, (54.5, 28), (66, 23.5), color=COLORS["green"], lw=2.4)

    save_figure(fig, "Fig_e024_02_comparator_fairness")


def figure_evidence_ladder() -> None:
    fig, ax = setup_canvas()
    add_title(
        fig,
        ax,
        "Evidence ladder and claim boundary",
        "A clean map of what the paper supports, what it does not support, and where S3 sits",
    )

    stair_colors = [
        (COLORS["blue_soft"], COLORS["blue"]),
        (COLORS["teal_soft"], COLORS["teal"]),
        (COLORS["green_soft"], COLORS["green"]),
        (COLORS["orange_soft"], COLORS["orange"]),
        ("#fff7df", COLORS["gold"]),
    ]
    stair_labels = [
        "traceable simulator",
        "canonical joint profile",
        "comparator discrimination",
        "robustness regime",
        "motif boundaries",
    ]
    x, y, w, h = 10, 18, 13, 10
    for idx, ((soft, edge), label) in enumerate(zip(stair_colors, stair_labels)):
        sx = x + idx * 12.5
        sy = y + idx * 9.5
        add_round_box(ax, sx, sy, w, h, fc=soft, ec=edge, lw=1.8)
        ax.text(sx + w / 2, sy + h / 2, label, ha="center", va="center", fontsize=11, fontweight="bold", color=COLORS["ink"])

    add_round_box(ax, 71, 58, 22, 18, fc="#eef5ff", ec=COLORS["navy"], lw=2.0)
    ax.text(82, 71, "Main supported claim", ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["navy"])
    ax.text(
        82,
        63,
        "replay-dependent slow\nbranch-level writing best\nexplains the joint simulator profile",
        ha="center",
        va="center",
        fontsize=11.3,
        color=COLORS["ink"],
    )
    add_arrow(ax, (60, 56), (71, 64), color=COLORS["navy"], lw=2.6)

    add_round_box(ax, 70, 22, 24, 24, fc=COLORS["red_soft"], ec=COLORS["red"], lw=1.8)
    ax.text(82, 41, "Not established", ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["red"])
    ax.text(
        82,
        30,
        "no unique molecular code\nno direct biological validation\nno direct measurement of $M_b$",
        ha="center",
        va="center",
        fontsize=11.2,
        color=COLORS["ink"],
    )

    add_round_box(ax, 6, 62, 22, 14, fc="#fffaf0", ec=COLORS["gold"], lw=1.6)
    ax.text(17, 72, "Supplementary only", ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["gold"])
    ax.text(
        17,
        66,
        "exploratory open-data\nbridges (S3)",
        ha="center",
        va="center",
        fontsize=11.2,
        color=COLORS["ink"],
    )
    add_arrow(ax, (28, 68), (44, 61), color=COLORS["gold"], lw=2.0, linestyle="--")

    ax.text(31, 13, "primary evidence ladder", fontsize=10.5, color=COLORS["gray"], fontweight="bold")
    ax.text(73, 14, "interpretive boundary", fontsize=10.5, color=COLORS["gray"], fontweight="bold")

    ax.add_patch(Rectangle((7, 11), 56, 2, facecolor="#dfe8f5", edgecolor="none", zorder=0))
    save_figure(fig, "Fig_e024_03_evidence_ladder")


def figure_overlap_damage_rescue() -> None:
    fig, ax = setup_canvas()
    add_title(
        fig,
        ax,
        "Overlap damage and targeted rescue logic",
        "A storyboard for why linking is fragile to focal overlap damage and preferentially rescued by overlap-targeted intervention",
    )

    panel_x = [4, 27.5, 51, 74.5]
    titles = [
        "A. Shared overlap branch",
        "B. After consolidation",
        "C. Focal damage",
        "D. Targeted rescue",
    ]
    header_colors = [COLORS["blue"], COLORS["green"], COLORS["red"], COLORS["orange"]]
    bg_colors = [COLORS["blue_soft"], COLORS["green_soft"], COLORS["red_soft"], COLORS["orange_soft"]]

    for idx, x0 in enumerate(panel_x):
        add_round_box(ax, x0, 12, 21, 74, fc=bg_colors[idx], ec="#d7dfec")
        add_section_header(ax, x0 + 2.0, 82, 17, titles[idx], header_colors[idx])
        draw_branch_schematic(ax, x0 + 10.5, 48, scale=0.55, highlight=True)

    ax.text(14.5, 67, "trace mu1", ha="center", fontsize=10.5, color=COLORS["blue"], fontweight="bold")
    ax.text(14.5, 63, "trace mu2", ha="center", fontsize=10.5, color=COLORS["orange"], fontweight="bold")
    ax.text(
        14.5,
        25,
        "two traces share b1\nbut also keep private branches",
        ha="center",
        fontsize=11,
        color=COLORS["ink"],
    )
    add_arrow(ax, (19, 52), (21.5, 52), color=COLORS["blue"], lw=2.2)

    ax.text(38, 67, r"$M_{b1}$ rises", ha="center", fontsize=15, fontweight="bold", color=COLORS["green"])
    ax.text(38, 61, "replay writes a persistent\nbias onto the overlap branch", ha="center", fontsize=10.8, color=COLORS["ink"])
    ax.text(38, 28, "linking increases after\nconsolidation", ha="center", fontsize=11.5, fontweight="bold", color=COLORS["green"])
    add_arrow(ax, (42.5, 52), (45, 52), color=COLORS["green"], lw=2.2)

    ax.text(61.5, 67, "damage to b1", ha="center", fontsize=14.5, fontweight="bold", color=COLORS["red"])
    ax.text(61.5, 61, "shared route is disrupted,\nprivate routes remain", ha="center", fontsize=10.8, color=COLORS["ink"])
    ax.text(61.5, 31, "linking drops strongly\nsingle-trace recall falls less", ha="center", fontsize=11.2, color=COLORS["red"], fontweight="bold")
    ax.plot([57.2, 65.8], [59.5, 44.5], color=COLORS["red"], linewidth=3)
    add_arrow(ax, (66.2, 52), (68.5, 52), color=COLORS["red"], lw=2.2)

    ax.text(85, 67, "rescue at overlap branch", ha="center", fontsize=14, fontweight="bold", color=COLORS["orange"])
    ax.text(85, 61, "restore the relevant branch-local\nstate, not just global gain", ha="center", fontsize=10.6, color=COLORS["ink"])
    ax.text(85, 28, "targeted rescue > generic rescue", ha="center", fontsize=11.5, fontweight="bold", color=COLORS["orange"])
    ax.add_patch(Circle((85, 46), 2.2, facecolor="#ffffff", edgecolor=COLORS["orange"], linewidth=2))
    ax.text(85, 46, "+", ha="center", va="center", fontsize=18, fontweight="bold", color=COLORS["orange"])

    save_figure(fig, "Fig_e024_04_overlap_damage_rescue_logic")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_biological_motivation()
    figure_comparator_fairness()
    figure_evidence_ladder()
    figure_overlap_damage_rescue()
    print(f"Generated concept figure pack in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
