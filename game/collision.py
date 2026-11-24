"""Strict AABB geometry: touching an edge is not overlap.

Collision resolution consults the level's broad-phase grid (``solids_near``)
with the player's AABB after moving in each axis; the resolution itself is the
same as C-07, so results are bit-identical to brute force over ``platforms``.
"""


def overlaps(a, b) -> bool:
    return (a.x < b.x + b.w and a.x + a.w > b.x
            and a.y < b.y + b.h and a.y + a.h > b.y)


def move_horizontal(player, level, width):
    # Preserve incoming direction until every simultaneous overlap is resolved.
    dx = player.vx
    player.x += dx
    for i in level.solids_near(player.x, player.y, player.w, player.h):
        solid = level.platforms[i]
        if overlaps(player, solid):
            if dx > 0:
                player.x = solid.x - player.w
            elif dx < 0:
                player.x = solid.x + solid.w
            player.vx = 0.0
    if player.x <= 0:
        player.x = 0.0
        if dx < 0:
            player.vx = 0.0
    elif player.x >= width - player.w:
        player.x = float(width - player.w)
        if dx > 0:
            player.vx = 0.0


def move_vertical(player, level):
    dy = player.vy
    player.grounded = False
    player.y += dy
    for i in level.solids_near(player.x, player.y, player.w, player.h):
        solid = level.platforms[i]
        if overlaps(player, solid):
            if dy > 0:
                player.y = solid.y - player.h
                player.grounded = True
            elif dy < 0:
                player.y = solid.y + solid.h
            player.vy = 0.0