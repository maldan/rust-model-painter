use mega_render::Handle;
use mega_render::Texture;

pub const TEX_SIZE: u32 = 2048;
/// Packed tangent-space flat normal (OpenGL / glTF): 128, 128, 255.
pub const FLAT_NORMAL: [f32; 4] = [128.0 / 255.0, 128.0 / 255.0, 1.0, 1.0];

/// Which material map the brush / layer stack targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PaintMap {
    #[default]
    Albedo,
    Metallic,
    Roughness,
    Normal,
}

impl PaintMap {
    pub const ALL: &'static [PaintMap] = &[
        PaintMap::Albedo,
        PaintMap::Metallic,
        PaintMap::Roughness,
        PaintMap::Normal,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PaintMap::Albedo => "Albedo",
            PaintMap::Metallic => "Metallic",
            PaintMap::Roughness => "Roughness",
            PaintMap::Normal => "Normal",
        }
    }

    pub fn index(self) -> usize {
        match self {
            PaintMap::Albedo => 0,
            PaintMap::Metallic => 1,
            PaintMap::Roughness => 2,
            PaintMap::Normal => 3,
        }
    }

    /// RGB write mask when packing this stack into the material map
    /// (glTF MR: G=roughness, B=metallic).
    pub fn channel_mask(self) -> [f32; 4] {
        match self {
            PaintMap::Albedo | PaintMap::Normal => [1.0, 1.0, 1.0, 1.0],
            PaintMap::Roughness => [0.0, 1.0, 0.0, 0.0],
            PaintMap::Metallic => [0.0, 0.0, 1.0, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayerKind {
    #[default]
    Paint,
    Fill,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PaintTarget {
    #[default]
    Content,
    Mask,
}

/// One stack layer — GPU texture is the source of truth for paint / mask pixels.
pub struct Layer {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub kind: LayerKind,
    /// Fill color (linear). Paint layers ignore this.
    pub fill: [f32; 4],
    pub tex: Option<Handle<Texture>>,
    /// Extra UDIM tiles after the first (`tex` is `udim_ids[0]`).
    pub extra_tex: Vec<(u32, Handle<Texture>)>,
    pub mask: Option<Handle<Texture>>,
    pub extra_mask: Vec<(u32, Handle<Texture>)>,
    /// Clear GPU paint tex to transparent before next composite.
    pub needs_clear: bool,
    /// Fill GPU mask with this RGBA8 before next composite (`None` = ready).
    pub mask_init: Option<[u8; 4]>,
}

impl Layer {
    pub fn paint(name: impl Into<String>, tex: Handle<Texture>) -> Self {
        Self {
            name: name.into(),
            visible: true,
            opacity: 1.0,
            kind: LayerKind::Paint,
            fill: [1.0, 1.0, 1.0, 1.0],
            tex: Some(tex),
            extra_tex: Vec::new(),
            mask: None,
            extra_mask: Vec::new(),
            needs_clear: true,
            mask_init: None,
        }
    }

    pub fn fill(name: impl Into<String>, color: [f32; 4]) -> Self {
        Self {
            name: name.into(),
            visible: true,
            opacity: 1.0,
            kind: LayerKind::Fill,
            fill: color,
            tex: None,
            extra_tex: Vec::new(),
            mask: None,
            extra_mask: Vec::new(),
            needs_clear: false,
            mask_init: None,
        }
    }

    pub fn gpu_handles(&self) -> impl Iterator<Item = Handle<Texture>> {
        self.tex
            .into_iter()
            .chain(self.extra_tex.iter().map(|(_, h)| *h))
            .chain(self.mask)
            .chain(self.extra_mask.iter().map(|(_, h)| *h))
    }

    pub fn content_tiles(&self, ids: &[u32]) -> Vec<(u32, Handle<Texture>)> {
        tiles_for(ids, self.tex, &self.extra_tex)
    }

    pub fn mask_tiles(&self, ids: &[u32]) -> Vec<(u32, Handle<Texture>)> {
        tiles_for(ids, self.mask, &self.extra_mask)
    }
}

/// Brush stroke mode — paint deposits color, erase removes layer coverage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PaintTool {
    #[default]
    Paint,
    Eraser,
}

impl PaintTool {
    pub const ALL: &'static [PaintTool] = &[PaintTool::Paint, PaintTool::Eraser];

    pub fn label(self) -> &'static str {
        match self {
            PaintTool::Paint => "Paint",
            PaintTool::Eraser => "Eraser",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Brush {
    pub color: [f32; 4],
    /// World-space radius.
    pub radius: f32,
    /// 0 = soft, 1 = hard.
    pub hardness: f32,
    pub opacity: f32,
    pub spacing: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            color: [0.9, 0.15, 0.12, 1.0],
            radius: 0.08,
            hardness: 0.55,
            opacity: 0.85,
            spacing: 0.35,
        }
    }
}

fn tiles_for(
    ids: &[u32],
    first: Option<Handle<Texture>>,
    extra: &[(u32, Handle<Texture>)],
) -> Vec<(u32, Handle<Texture>)> {
    let Some(h0) = first else {
        return extra.to_vec();
    };
    let Some(&id0) = ids.first() else {
        return vec![(1001, h0)];
    };
    let mut out = vec![(id0, h0)];
    out.extend_from_slice(extra);
    out
}

pub fn luma(rgba: [f32; 4]) -> f32 {
    0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
}

pub struct PaintDocument {
    pub width: u32,
    pub height: u32,
    /// Underpaint when all layers are transparent (linear 0..1).
    pub base_rgba: [f32; 4],
    pub layers: Vec<Layer>,
    pub active: usize,
    pub paint_target: PaintTarget,
    pub composite_dirty: bool,
}

impl PaintDocument {
    pub fn new(width: u32, height: u32, base_rgba: [f32; 4]) -> Self {
        Self {
            width,
            height,
            base_rgba,
            layers: Vec::new(),
            active: 0,
            paint_target: PaintTarget::Content,
            composite_dirty: true,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.composite_dirty = true;
    }

    pub fn take_composite_dirty(&mut self) -> bool {
        let d = self.composite_dirty;
        self.composite_dirty = false;
        d
    }

    pub fn end_stroke(&mut self) {}

    pub fn active_layer(&self) -> Option<&Layer> {
        self.layers.get(self.active)
    }

    pub fn clamp_paint_target(&mut self) {
        let ok = self
            .active_layer()
            .is_some_and(|l| l.mask.is_some() || self.paint_target == PaintTarget::Content);
        if !ok {
            self.paint_target = PaintTarget::Content;
        }
    }

    pub fn move_active(&mut self, delta: isize) {
        let n = self.layers.len() as isize;
        if n <= 1 {
            return;
        }
        let to = (self.active as isize + delta).clamp(0, n - 1) as usize;
        if to == self.active {
            return;
        }
        self.layers.swap(self.active, to);
        self.active = to;
        self.mark_dirty();
    }
}
