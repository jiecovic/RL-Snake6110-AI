// rust/src/engine/vocab.rs

#[derive(Copy, Clone)]
pub struct VocabClass {
    pub name: &'static str,
    pub members: &'static [&'static str],
}

#[derive(Copy, Clone)]
pub struct VocabDef {
    pub name: &'static str,
    pub classes: &'static [VocabClass],
}

fn normalize_vocab_name(name: &str) -> String {
    let key = name.trim();
    if let Some(rest) = key.strip_prefix("pov_") {
        return format!("head_{}", rest);
    }
    if let Some(rest) = key.strip_prefix("global_") {
        return format!("world_{}", rest);
    }
    key.to_string()
}

static GLOBAL_NO_BORDER_V1: [VocabClass; 9] = [
    VocabClass {
        name: "OOB",
        members: &["OOB"],
    },
    VocabClass {
        name: "EMPTY",
        members: &[
            "EMPTY",
            "WALL_TL",
            "WALL_TR",
            "WALL_BL",
            "WALL_BR",
            "WALL_TOP",
            "WALL_BOTTOM",
            "WALL_LEFT",
            "WALL_RIGHT",
        ],
    },
    VocabClass {
        name: "FOOD",
        members: &["FOOD"],
    },
    VocabClass {
        name: "HEAD_UP",
        members: &["SNAKE_HEAD_UP"],
    },
    VocabClass {
        name: "HEAD_DOWN",
        members: &["SNAKE_HEAD_DOWN"],
    },
    VocabClass {
        name: "HEAD_LEFT",
        members: &["SNAKE_HEAD_LEFT"],
    },
    VocabClass {
        name: "HEAD_RIGHT",
        members: &["SNAKE_HEAD_RIGHT"],
    },
    VocabClass {
        name: "BODY",
        members: &[
            "SNAKE_BODY_VERTICAL_UP",
            "SNAKE_BODY_VERTICAL_DOWN",
            "SNAKE_BODY_HORIZONTAL_LEFT",
            "SNAKE_BODY_HORIZONTAL_RIGHT",
            "SNAKE_BODY_BR",
            "SNAKE_BODY_BL",
            "SNAKE_BODY_TR",
            "SNAKE_BODY_TL",
        ],
    },
    VocabClass {
        name: "TAIL",
        members: &[
            "SNAKE_TAIL_UP",
            "SNAKE_TAIL_DOWN",
            "SNAKE_TAIL_LEFT",
            "SNAKE_TAIL_RIGHT",
        ],
    },
];

static GLOBAL_V2: [VocabClass; 19] = [
    VocabClass {
        name: "OOB",
        members: &["OOB"],
    },
    VocabClass {
        name: "EMPTY",
        members: &[
            "EMPTY",
            "WALL_TL",
            "WALL_TR",
            "WALL_BL",
            "WALL_BR",
            "WALL_TOP",
            "WALL_BOTTOM",
            "WALL_LEFT",
            "WALL_RIGHT",
        ],
    },
    VocabClass {
        name: "FOOD",
        members: &["FOOD"],
    },
    VocabClass {
        name: "HEAD_UP",
        members: &["SNAKE_HEAD_UP"],
    },
    VocabClass {
        name: "HEAD_DOWN",
        members: &["SNAKE_HEAD_DOWN"],
    },
    VocabClass {
        name: "HEAD_LEFT",
        members: &["SNAKE_HEAD_LEFT"],
    },
    VocabClass {
        name: "HEAD_RIGHT",
        members: &["SNAKE_HEAD_RIGHT"],
    },
    VocabClass {
        name: "BODY_VERTICAL_UP",
        members: &["SNAKE_BODY_VERTICAL_UP"],
    },
    VocabClass {
        name: "BODY_VERTICAL_DOWN",
        members: &["SNAKE_BODY_VERTICAL_DOWN"],
    },
    VocabClass {
        name: "BODY_HORIZONTAL_LEFT",
        members: &["SNAKE_BODY_HORIZONTAL_LEFT"],
    },
    VocabClass {
        name: "BODY_HORIZONTAL_RIGHT",
        members: &["SNAKE_BODY_HORIZONTAL_RIGHT"],
    },
    VocabClass {
        name: "BODY_BR",
        members: &["SNAKE_BODY_BR"],
    },
    VocabClass {
        name: "BODY_BL",
        members: &["SNAKE_BODY_BL"],
    },
    VocabClass {
        name: "BODY_TR",
        members: &["SNAKE_BODY_TR"],
    },
    VocabClass {
        name: "BODY_TL",
        members: &["SNAKE_BODY_TL"],
    },
    VocabClass {
        name: "TAIL_UP",
        members: &["SNAKE_TAIL_UP"],
    },
    VocabClass {
        name: "TAIL_DOWN",
        members: &["SNAKE_TAIL_DOWN"],
    },
    VocabClass {
        name: "TAIL_LEFT",
        members: &["SNAKE_TAIL_LEFT"],
    },
    VocabClass {
        name: "TAIL_RIGHT",
        members: &["SNAKE_TAIL_RIGHT"],
    },
];

static POV_V1: [VocabClass; 7] = [
    VocabClass {
        name: "OOB",
        members: &["OOB"],
    },
    VocabClass {
        name: "EMPTY",
        members: &["EMPTY"],
    },
    VocabClass {
        name: "WALL",
        members: &[
            "WALL_TL",
            "WALL_TR",
            "WALL_BL",
            "WALL_BR",
            "WALL_TOP",
            "WALL_BOTTOM",
            "WALL_LEFT",
            "WALL_RIGHT",
        ],
    },
    VocabClass {
        name: "FOOD",
        members: &["FOOD"],
    },
    VocabClass {
        name: "HEAD",
        members: &[
            "SNAKE_HEAD_UP",
            "SNAKE_HEAD_DOWN",
            "SNAKE_HEAD_LEFT",
            "SNAKE_HEAD_RIGHT",
        ],
    },
    VocabClass {
        name: "BODY",
        members: &[
            "SNAKE_BODY_VERTICAL_UP",
            "SNAKE_BODY_VERTICAL_DOWN",
            "SNAKE_BODY_HORIZONTAL_LEFT",
            "SNAKE_BODY_HORIZONTAL_RIGHT",
            "SNAKE_BODY_BR",
            "SNAKE_BODY_BL",
            "SNAKE_BODY_TR",
            "SNAKE_BODY_TL",
        ],
    },
    VocabClass {
        name: "TAIL",
        members: &[
            "SNAKE_TAIL_UP",
            "SNAKE_TAIL_DOWN",
            "SNAKE_TAIL_LEFT",
            "SNAKE_TAIL_RIGHT",
        ],
    },
];

static POV_V2: [VocabClass; 10] = [
    VocabClass {
        name: "OOB",
        members: &["OOB"],
    },
    VocabClass {
        name: "EMPTY",
        members: &["EMPTY"],
    },
    VocabClass {
        name: "WALL",
        members: &[
            "WALL_TL",
            "WALL_TR",
            "WALL_BL",
            "WALL_BR",
            "WALL_TOP",
            "WALL_BOTTOM",
            "WALL_LEFT",
            "WALL_RIGHT",
        ],
    },
    VocabClass {
        name: "FOOD",
        members: &["FOOD"],
    },
    VocabClass {
        name: "HEAD_UP",
        members: &["SNAKE_HEAD_UP"],
    },
    VocabClass {
        name: "HEAD_DOWN",
        members: &["SNAKE_HEAD_DOWN"],
    },
    VocabClass {
        name: "HEAD_LEFT",
        members: &["SNAKE_HEAD_LEFT"],
    },
    VocabClass {
        name: "HEAD_RIGHT",
        members: &["SNAKE_HEAD_RIGHT"],
    },
    VocabClass {
        name: "BODY",
        members: &[
            "SNAKE_BODY_VERTICAL_UP",
            "SNAKE_BODY_VERTICAL_DOWN",
            "SNAKE_BODY_HORIZONTAL_LEFT",
            "SNAKE_BODY_HORIZONTAL_RIGHT",
            "SNAKE_BODY_BR",
            "SNAKE_BODY_BL",
            "SNAKE_BODY_TR",
            "SNAKE_BODY_TL",
        ],
    },
    VocabClass {
        name: "TAIL",
        members: &[
            "SNAKE_TAIL_UP",
            "SNAKE_TAIL_DOWN",
            "SNAKE_TAIL_LEFT",
            "SNAKE_TAIL_RIGHT",
        ],
    },
];

static COARSE_V1: [VocabClass; 10] = [
    VocabClass {
        name: "OOB",
        members: &["OOB"],
    },
    VocabClass {
        name: "EMPTY",
        members: &["EMPTY"],
    },
    VocabClass {
        name: "WALL",
        members: &[
            "WALL_TL",
            "WALL_TR",
            "WALL_BL",
            "WALL_BR",
            "WALL_TOP",
            "WALL_BOTTOM",
            "WALL_LEFT",
            "WALL_RIGHT",
        ],
    },
    VocabClass {
        name: "FOOD",
        members: &["FOOD"],
    },
    VocabClass {
        name: "HEAD_UP",
        members: &["SNAKE_HEAD_UP"],
    },
    VocabClass {
        name: "HEAD_DOWN",
        members: &["SNAKE_HEAD_DOWN"],
    },
    VocabClass {
        name: "HEAD_LEFT",
        members: &["SNAKE_HEAD_LEFT"],
    },
    VocabClass {
        name: "HEAD_RIGHT",
        members: &["SNAKE_HEAD_RIGHT"],
    },
    VocabClass {
        name: "BODY",
        members: &[
            "SNAKE_BODY_VERTICAL_UP",
            "SNAKE_BODY_VERTICAL_DOWN",
            "SNAKE_BODY_HORIZONTAL_LEFT",
            "SNAKE_BODY_HORIZONTAL_RIGHT",
            "SNAKE_BODY_BR",
            "SNAKE_BODY_BL",
            "SNAKE_BODY_TR",
            "SNAKE_BODY_TL",
        ],
    },
    VocabClass {
        name: "TAIL",
        members: &[
            "SNAKE_TAIL_UP",
            "SNAKE_TAIL_DOWN",
            "SNAKE_TAIL_LEFT",
            "SNAKE_TAIL_RIGHT",
        ],
    },
];

static VOCAB_DEFS: [VocabDef; 5] = [
    VocabDef {
        name: "world_no_border_v1",
        classes: &GLOBAL_NO_BORDER_V1,
    },
    VocabDef {
        name: "world_v2",
        classes: &GLOBAL_V2,
    },
    VocabDef {
        name: "head_v1",
        classes: &POV_V1,
    },
    VocabDef {
        name: "head_v2",
        classes: &POV_V2,
    },
    VocabDef {
        name: "coarse_v1",
        classes: &COARSE_V1,
    },
];

pub fn vocab_defs() -> &'static [VocabDef] {
    &VOCAB_DEFS
}

fn vocab_def_by_name(name: &str) -> Option<&'static VocabDef> {
    let key = normalize_vocab_name(name);
    vocab_defs().iter().find(|d| d.name == key)
}

fn vocab_err_unknown(name: &str) -> String {
    let mut names: Vec<&str> = vocab_defs().iter().map(|d| d.name).collect();
    names.sort();
    let preview = names.iter().take(20).cloned().collect::<Vec<_>>().join(", ");
    let more = if names.len() > 20 {
        format!(" ... (+{} more)", names.len() - 20)
    } else {
        String::new()
    };
    format!("Unknown tile_vocab={:?}. Available: {}{}", name, preview, more)
}

fn tile_id_for_name(name: &str) -> Option<u8> {
    let names = crate::engine::tileset::tileset_tile_names_raw();
    for (i, n) in names.iter().enumerate() {
        if *n == name {
            return Some(i as u8);
        }
    }
    None
}

fn build_lut(def: &VocabDef) -> Result<Vec<u8>, String> {
    let raw_size = crate::engine::tileset::tileset_tile_count();
    let class_count = def.classes.len();
    if class_count > u8::MAX as usize {
        return Err("tile_vocab has too many classes for u8 ids".to_string());
    }

    let mut lut = vec![0u8; raw_size];
    let mut seen = vec![false; raw_size];

    for (class_id, class_def) in def.classes.iter().enumerate() {
        let cid = class_id as u8;
        for member in class_def.members.iter() {
            let tid = tile_id_for_name(member).ok_or_else(|| {
                let valid = crate::engine::tileset::tileset_tile_names()
                    .join(", ");
                format!(
                    "Unknown tile name {:?} in class {:?}. Valid tiles: {}",
                    member, class_def.name, valid
                )
            })? as usize;

            if seen[tid] {
                return Err(format!(
                    "Tile id {} appears in multiple classes (latest: {:?})",
                    tid, class_def.name
                ));
            }
            seen[tid] = true;
            lut[tid] = cid;
        }
    }

    let mut missing: Vec<&'static str> = Vec::new();
    for (i, seen_i) in seen.iter().enumerate() {
        if !*seen_i {
            let name = crate::engine::tileset::tileset_tile_names_raw()[i];
            missing.push(name);
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "Tile vocab is missing tiles: {}",
            missing.join(", ")
        ));
    }

    Ok(lut)
}

pub fn vocab_num_classes(name: &str) -> Result<usize, String> {
    let def = vocab_def_by_name(name).ok_or_else(|| vocab_err_unknown(name))?;
    Ok(def.classes.len())
}

pub fn vocab_class_names(name: &str) -> Result<Vec<String>, String> {
    let def = vocab_def_by_name(name).ok_or_else(|| vocab_err_unknown(name))?;
    Ok(def.classes.iter().map(|c| c.name.to_string()).collect())
}

pub fn vocab_classes(name: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    let def = vocab_def_by_name(name).ok_or_else(|| vocab_err_unknown(name))?;
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for class_def in def.classes.iter() {
        let members = class_def.members.iter().map(|m| m.to_string()).collect();
        out.push((class_def.name.to_string(), members));
    }
    Ok(out)
}

pub fn vocab_lut(name: &str) -> Result<Vec<u8>, String> {
    let def = vocab_def_by_name(name).ok_or_else(|| vocab_err_unknown(name))?;
    build_lut(def)
}
