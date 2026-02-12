# src/snake_rl/game/rendering/pygame/layout.py

from __future__ import annotations

from snake_rl.game.rendering.pygame.window import PygameRenderContext, Rect


def hud_layout(ctx: PygameRenderContext) -> dict[str, Rect]:
    hud = ctx.hud_panel
    pad_x = int(ctx.layout.hud_padding_x)
    pad_y = int(ctx.layout.hud_padding_y)
    col_gap = int(ctx.layout.hud_col_gap)
    control_h = ctx.label_font.get_height() + 6

    inner_w = hud.w - 2 * pad_x - 3 * col_gap
    inner_h = hud.h - 2 * pad_y - control_h - int(ctx.layout.panel_gap)

    w_status = int(ctx.layout.hud_col_w_status)
    w_perf = int(ctx.layout.hud_col_w_perf)
    w_info = int(ctx.layout.hud_col_w_info)
    min_w = max(80, int(ctx.layout.hud_col_min))
    min_last = max(120, int(ctx.layout.hud_col_last_min))

    cols = [w_status, w_perf, w_info]
    last = inner_w - sum(cols)
    if last < min_last:
        deficit = min_last - last
        for i in (2, 1, 0):
            if deficit <= 0:
                break
            reducible = max(0, cols[i] - min_w)
            take = min(deficit, reducible)
            cols[i] -= take
            deficit -= take
        last = inner_w - sum(cols)
    if last < min_last:
        if inner_w < 4 * min_w:
            w = inner_w // 4
            cols = [w, w, w]
            last = inner_w - 3 * w
        else:
            cols = [min_w, min_w, min_w]
            last = inner_w - 3 * min_w
    cols.append(last)

    x0 = int(hud.x + pad_x)
    top_y = int(hud.y + pad_y)
    col_x = [x0]
    for w in cols[:-1]:
        col_x.append(col_x[-1] + int(w) + col_gap)

    status = Rect(x=col_x[0], y=top_y, w=int(cols[0]), h=inner_h)
    perf = Rect(x=col_x[1], y=top_y, w=int(cols[1]), h=inner_h)
    info = Rect(x=col_x[2], y=top_y, w=int(cols[2]), h=inner_h)
    features = Rect(x=col_x[3], y=top_y, w=int(cols[3]), h=inner_h)

    return {
        "status": status,
        "perf": perf,
        "info": info,
        "features": features,
    }
