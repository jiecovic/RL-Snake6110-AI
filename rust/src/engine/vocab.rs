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

static GLOBAL_NO_BORDER_V1: [VocabClass; 8] = [
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

static GLOBAL_V2: [VocabClass; 18] = [
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

static POV_V1: [VocabClass; 6] = [
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

static POV_V2: [VocabClass; 9] = [
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

static COARSE_V1: [VocabClass; 9] = [
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
        name: "global_no_border_v1",
        classes: &GLOBAL_NO_BORDER_V1,
    },
    VocabDef {
        name: "global_v2",
        classes: &GLOBAL_V2,
    },
    VocabDef {
        name: "pov_v1",
        classes: &POV_V1,
    },
    VocabDef {
        name: "pov_v2",
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
