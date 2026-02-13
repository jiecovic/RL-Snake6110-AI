# src/snake_rl/game/rendering/pygame/layout.py

from __future__ import annotations

from snake_rl.game.rendering.pygame.window import PygameRenderContext, Rect


def hud_layout(ctx: PygameRenderContext) -> dict[str, Rect]:
    hud = ctx.hud_panel
    pad_x = int(ctx.layout.hud_padding_x)
    pad_y = int(ctx.layout.hud_padding_y)
    col_gap = int(ctx.layout.hud_col_gap)
    row_gap = int(ctx.layout.hud_row_gap)
    control_h = ctx.label_font.get_height() + 6

    layout_mode = str(ctx.layout.hud_layout).strip().lower()
    extended = layout_mode == "extended"
    grid2 = layout_mode in {"grid2", "grid2_right", "right_grid2"}
    num_cols = 2 if grid2 else (6 if extended else 4)

    inner_w = hud.w - 2 * pad_x - (num_cols - 1) * col_gap
    inner_h = hud.h - 2 * pad_y - control_h - int(ctx.layout.panel_gap)

    w_status = int(ctx.layout.hud_col_w_status)
    w_perf = int(ctx.layout.hud_col_w_perf)
    w_info = int(ctx.layout.hud_col_w_info)
    w_stats = int(ctx.layout.hud_col_w_stats)
    w_episode = int(ctx.layout.hud_col_w_episode)
    min_w = max(80, int(ctx.layout.hud_col_min))
    min_last = max(120, int(ctx.layout.hud_col_last_min))

    if grid2:
        left_pref = max(w_status, w_info, w_episode)
        if inner_w < min_w * 2:
            w = inner_w // 2
            cols = [w]
            last = inner_w - w
        else:
            w_left = min(left_pref, inner_w - min_w)
            w_right = inner_w - w_left
            if w_right < min_w:
                w_right = min_w
                w_left = inner_w - w_right
            cols = [w_left]
            last = w_right
    elif extended:
        cols = [w_status, w_perf, w_info, w_stats, w_episode]
    else:
        cols = [w_status, w_perf, w_info]
    last = inner_w - sum(cols)
    if not grid2 and last < min_last:
        deficit = min_last - last
        for i in reversed(range(len(cols))):
            if deficit <= 0:
                break
            reducible = max(0, cols[i] - min_w)
            take = min(deficit, reducible)
            cols[i] -= take
            deficit -= take
        last = inner_w - sum(cols)
    if not grid2 and last < min_last:
        if inner_w < num_cols * min_w:
            w = inner_w // num_cols
            cols = [w for _ in range(num_cols - 1)]
            last = inner_w - (num_cols - 1) * w
        else:
            cols = [min_w for _ in range(num_cols - 1)]
            last = inner_w - (num_cols - 1) * min_w
    cols.append(last)

    x0 = int(hud.x + pad_x)
    top_y = int(hud.y + pad_y)
    col_x = [x0]
    for w in cols[:-1]:
        col_x.append(col_x[-1] + int(w) + col_gap)

    if grid2:
        if ctx.hud_row_heights is not None:
            h1, h2, h3 = (max(0, int(v)) for v in ctx.hud_row_heights)
            total = h1 + h2 + h3 + 2 * row_gap
            if total > inner_h and total > 0:
                scale = inner_h / float(total)
                h1 = max(1, int(h1 * scale))
                h2 = max(1, int(h2 * scale))
                h3 = max(1, inner_h - 2 * row_gap - h1 - h2)
            else:
                h3 = max(1, inner_h - 2 * row_gap - h1 - h2)
            row_h = h1
            row2_h = h2
            last_h = h3
        else:
            row_h = max(0, (inner_h - 2 * row_gap) // 3)
            row2_h = row_h
            last_h = max(0, inner_h - 2 * row_gap - 2 * row_h)
        row_y = [
            top_y,
            top_y + row_h + row_gap,
            top_y + row_h + row_gap + row2_h + row_gap,
        ]
        status = Rect(x=col_x[0], y=row_y[0], w=int(cols[0]), h=row_h)
        perf = Rect(x=col_x[1], y=row_y[0], w=int(cols[1]), h=row_h)
        info = Rect(x=col_x[0], y=row_y[1], w=int(cols[0]), h=row2_h)
        stats = Rect(x=col_x[1], y=row_y[1], w=int(cols[1]), h=row2_h)
        episode = Rect(x=col_x[0], y=row_y[2], w=int(cols[0]), h=last_h)
        features = Rect(x=col_x[1], y=row_y[2], w=int(cols[1]), h=last_h)
        return {
            "status": status,
            "perf": perf,
            "info": info,
            "stats": stats,
            "episode": episode,
            "features": features,
        }

    status = Rect(x=col_x[0], y=top_y, w=int(cols[0]), h=inner_h)
    perf = Rect(x=col_x[1], y=top_y, w=int(cols[1]), h=inner_h)
    info = Rect(x=col_x[2], y=top_y, w=int(cols[2]), h=inner_h)
    if extended:
        stats = Rect(x=col_x[3], y=top_y, w=int(cols[3]), h=inner_h)
        episode = Rect(x=col_x[4], y=top_y, w=int(cols[4]), h=inner_h)
        features = Rect(x=col_x[5], y=top_y, w=int(cols[5]), h=inner_h)
        return {
            "status": status,
            "perf": perf,
            "info": info,
            "stats": stats,
            "episode": episode,
            "features": features,
        }
    features = Rect(x=col_x[3], y=top_y, w=int(cols[3]), h=inner_h)
    return {
        "status": status,
        "perf": perf,
        "info": info,
        "features": features,
    }
