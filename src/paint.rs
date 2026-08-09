use mega_render::Handle;
use mega_render::Texture;

pub const TEX_SIZE: u32 = 1024;

/// Which material map the brush / layer stack targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PaintMap {
    #[default]
    Albedo,
    Metallic,
    Roughness,
}

impl PaintMap {
    pub const ALL: &'static [PaintMap] =
        &[PaintMap::Albedo, PaintMap::Metallic, PaintMap::Roughness];

    pub fn label(self) -> &'static str {
        match self {
            PaintMap::Albedo => "Albedo",
            PaintMap::Metallic => "Metallic",
            PaintMap::Roughness => "Roughness",
        }
    }

    pub fn index(self) -> usize {
        match self {
            PaintMap::Albedo => 0,
            PaintMap::Metallic => 1,
            PaintMap::Roughness => 2,
        }
    }

    /// RGB write mask when packing this stack into the material map
    /// (glTF MR: G=roughness, B=metallic).
    pub fn channel_mask(self) -> [f32; 4] {
        match self {
            PaintMap::Albedo => [1.0, 1.0, 1.0, 1.0],
            PaintMap::Roughness => [0.0, 1.0, 0.0, 0.0],
            PaintMap::Metallic => [0.0, 0.0, 1.0, 0.0],
        }
    }
}

/// One paint layer — GPU texture is the source of truth.
pub struct Layer {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub tex: Handle<Texture>,
    /// Clear GPU tex to transparent before next composite.
    pub needs_clear: bool,
}

impl Layer {
    pub fn new(name: impl Into<String>, tex: Handle<Texture>) -> Self {
        Self {
            name: name.into(),
            visible: true,
            opacity: 1.0,
            tex,
            // gpu_resident tex is uninitialized until cleared.
            needs_clear: true,
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

pub struct PaintDocument {
    pub width: u32,
    pub height: u32,
    /// Underpaint when all layers are transparent (linear 0..1).
    pub base_rgba: [f32; 4],
    pub layers: Vec<Layer>,
    pub active: usize,
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
