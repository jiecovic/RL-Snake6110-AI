// rust/src/engine/spatial/geometry.rs
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

#[inline]
pub fn idx(x: i32, y: i32, width: usize) -> usize {
    (y as usize) * width + (x as usize)
}

#[inline]
pub fn dir_turn_left(d: i8) -> i8 {
    (d + 3) % 4
}

#[inline]
pub fn dir_turn_right(d: i8) -> i8 {
    (d + 1) % 4
}

#[inline]
pub fn dir_vec(d: i8) -> (i32, i32) {
    match d {
        0 => (0, -1),
        1 => (1, 0),
        2 => (0, 1),
        3 => (-1, 0),
        _ => (0, 0),
    }
}

pub fn compute_spawn_cells(head: Point, length: usize, direction: i8) -> Vec<Point> {
    let (dx, dy) = dir_vec(direction);
    let mut out: Vec<Point> = Vec::with_capacity(length);
    for i in 0..length {
        out.push(Point {
            x: head.x - dx * (i as i32),
            y: head.y - dy * (i as i32),
        });
    }
    out
}

pub fn is_straight_spawn_valid(
    cells: &[Point],
    width: i32,
    height: i32,
    wall_mask: &[u8],
) -> bool {
    for p in cells.iter() {
        if p.x < 0 || p.x >= width || p.y < 0 || p.y >= height {
            return false;
        }
        if p.x == 0 || p.x == width - 1 || p.y == 0 || p.y == height - 1 {
            return false;
        }
        let i = idx(p.x, p.y, width as usize);
        if wall_mask[i] != 0 {
            return false;
        }
    }
    true
}

#[inline]
pub fn idx_to_point(i: usize, width: usize) -> Point {
    Point {
        x: (i % width) as i32,
        y: (i / width) as i32,
    }
}
